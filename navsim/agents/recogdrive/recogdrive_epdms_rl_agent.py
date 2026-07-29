"""GRPO RL agent with the EPDMS (navsim_v2) reward + two-frame EC term.

Differences to the stock v1-PDMS GRPO path (all measured on the four bucket
experts, see project notes):

* reward = official EPDMS assembled from the ported v2 scorer
  (NC*DAC*DDC*TLC multiplicative; EP5/TTC5/LK2/HC2 + EC2 weighted). navtrain
  samples arrive as adjacent-frame pairs so EC is computed exactly like the
  official ``SceneAggregator``; SimScale samples have no adjacent frame, so EC
  is dropped from their weight normalization (official NaN convention) and the
  v1-PDMS reward is used there (their metric caches are v1-format).
* every rollout's EC uses the partner frame's DETERMINISTIC plan as the fixed
  reference (reward of rollout g depends only on rollout g).
* per-expert scene bonus on non-saturated quantities (ep / ec / gate).
* degenerate-group filtering: groups whose reward std collapses produce zero
  advantage instead of amplified noise, and the loss is renormalized over the
  surviving rollouts.
* LR schedule: monotone warmup-cosine over the REAL number of epochs (the base
  class hardcodes ``epochs=10``, which oscillates for longer runs).
"""

from __future__ import annotations

import lzma
import pickle
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.epdms import (
    MetricCacheIndexV2,
    build_v2_simulator_and_scorer,
    epdms_score,
    score_token_proposals_v2,
    two_frame_extended_comfort,
)
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.utils.lr_scheduler import WarmupCosLR
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score as pdm_score_v1

SCENE_TERMS = ("none", "ep", "ec", "gate")


class ReCogDriveEpdmsRLAgent(ReCogDriveAgent):
    def __init__(
        self,
        *args: Any,
        epdms_metric_cache_v2_path: str = "",
        scene_term: str = "none",
        scene_term_weight: float = 0.3,
        rl_sample_time: int = 8,
        rl_bc_coeff: float = 0.1,
        degenerate_std_threshold: float = 1e-3,
        rl_max_epochs: int = 15,
        rl_warmup_epochs: int = 2,
        rl_min_lr_ratio: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.grpo:
            raise ValueError("ReCogDriveEpdmsRLAgent requires grpo=True.")
        if scene_term not in SCENE_TERMS:
            raise ValueError(f"scene_term must be one of {SCENE_TERMS}, got {scene_term!r}")
        if not epdms_metric_cache_v2_path:
            raise ValueError("epdms_metric_cache_v2_path (navtrain v2 metric cache) is required.")

        self._epdms_cache_path = epdms_metric_cache_v2_path
        self._scene_term = scene_term
        self._scene_term_weight = float(scene_term_weight)
        self._sample_time = int(rl_sample_time)
        self._bc_coeff = float(rl_bc_coeff)
        self._degenerate_std_threshold = float(degenerate_std_threshold)
        self._rl_max_epochs = int(rl_max_epochs)
        self._rl_warmup_epochs = int(rl_warmup_epochs)
        self._rl_min_lr_ratio = float(rl_min_lr_ratio)

        self._v2_index: Optional[MetricCacheIndexV2] = None
        self._v2_simulator = None
        self._v2_scorer = None

    # ------------------------------------------------------------------ setup

    def initialize(self) -> None:
        super().initialize()
        self._v2_index = MetricCacheIndexV2(self._epdms_cache_path)
        if len(self._v2_index) == 0:
            raise RuntimeError(f"empty v2 metric cache index at {self._epdms_cache_path}")
        self._v2_simulator, self._v2_scorer = build_v2_simulator_and_scorer()
        print(
            f"[EpdmsRL] v2 metric cache: {len(self._v2_index)} tokens | scene_term={self._scene_term}"
            f"(w={self._scene_term_weight}) | G={self._sample_time} bc={self._bc_coeff}"
        )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        params = [p for p in self.action_head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95))
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=self._lr * self._rl_min_lr_ratio,
            epochs=self._rl_max_epochs,
            warmup_epochs=self._rl_warmup_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ---------------------------------------------------------------- forward

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None):
        if not (self.training and self.grpo):
            return super().forward(features, targets, tokens_list)

        model_dtype = next(self.action_head.parameters()).dtype
        last_hidden_state = features["last_hidden_state"].cuda().to(model_dtype)
        history_trajectory = features["history_trajectory"].cuda()
        status_feature = features["status_feature"].cuda()
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)
        his_traj = history_trajectory.view(history_trajectory.size(0), -1).to(model_dtype)
        status_feature = status_feature.to(model_dtype)

        num_pairs = int(features.get("epdms_num_pairs", torch.tensor(0)).item())
        pair_dt = features.get("epdms_pair_dt", torch.tensor([], dtype=torch.float64)).cpu().numpy()

        return self._forward_grpo_epdms(
            last_hidden_state, his_traj, status_feature, list(tokens_list), num_pairs, pair_dt
        )

    # ----------------------------------------------------------------- reward

    def _score_navtrain(self, token: str, proposals: np.ndarray):
        metric_cache = self._v2_index.load(token)
        return score_token_proposals_v2(metric_cache, proposals, self._v2_simulator, self._v2_scorer)

    def _score_simscale(self, token: str, rollouts: np.ndarray) -> Dict[str, np.ndarray]:
        planner = self.action_head
        path = planner.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(path, "rb") as f:
            metric_cache = pickle.load(f)
        out = {"score": [], "ego_progress": [], "gate": []}
        for poses in rollouts:
            res = asdict(
                pdm_score_v1(
                    metric_cache=metric_cache,
                    model_trajectory=Trajectory(np.asarray(poses, dtype=np.float32)),
                    future_sampling=planner.simulator.proposal_sampling,
                    simulator=planner.simulator,
                    scorer=planner.train_scorer,
                )
            )
            out["score"].append(float(res["score"]))
            out["ego_progress"].append(float(res["ego_progress"]))
            out["gate"].append(float(res["no_at_fault_collisions"]) * float(res["drivable_area_compliance"]))
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def _scene_bonus(self, base: np.ndarray, ep: np.ndarray, ec: Optional[np.ndarray], gate: np.ndarray) -> np.ndarray:
        if self._scene_term == "ep":
            return base + self._scene_term_weight * ep
        if self._scene_term == "gate":
            return base + self._scene_term_weight * gate
        if self._scene_term == "ec":
            if ec is None:
                return base
            return base + self._scene_term_weight * ec
        return base

    # ------------------------------------------------------------------- grpo

    def _forward_grpo_epdms(
        self,
        vl_features: torch.Tensor,   # (B, L, D)
        his_traj: torch.Tensor,      # (B, 12)
        status_feature: torch.Tensor,
        tokens: List[str],
        num_pairs: int,
        pair_dt: np.ndarray,
    ) -> BatchFeature:
        planner = self.action_head
        if hasattr(planner, "set_frozen_modules_to_eval_mode"):
            planner.set_frozen_modules_to_eval_mode()
        B, G = vl_features.shape[0], self._sample_time
        n_nav = 2 * num_pairs

        # deterministic plans (fixed EC references), no grad
        with torch.no_grad():
            _, det_trajs = planner.sample_chain(vl_features, his_traj, status_feature, deterministic=True)
        det_np = det_trajs.float().cpu().numpy()

        # stochastic rollouts
        vl_rep = vl_features.repeat_interleave(G, 0)
        his_rep = his_traj.repeat_interleave(G, 0)
        status_rep = status_feature.repeat_interleave(G, 0)
        chains, trajs = planner.sample_chain(vl_rep, his_rep, status_rep, deterministic=False)
        trajs_np = trajs.float().cpu().numpy().reshape(B, G, *trajs.shape[1:])

        rewards = np.zeros((B, G), dtype=np.float64)
        det_states: Dict[int, np.ndarray] = {}
        nav_scores: Dict[int, Any] = {}

        # -- navtrain (v2 EPDMS): score [det, rollouts] in one batched call
        for i in range(n_nav):
            proposals = np.concatenate([det_np[i : i + 1], trajs_np[i]], axis=0)  # (1+G, H, 3)
            out = self._score_navtrain(tokens[i], proposals)
            det_states[i] = out.simulated_states[0]
            nav_scores[i] = out

        for p in range(num_pairs):
            prev_i, cur_i = 2 * p, 2 * p + 1
            dt = float(pair_dt[p])
            for i, partner in ((cur_i, prev_i), (prev_i, cur_i)):
                out = nav_scores[i]
                roll_sub = {k: v[1:] for k, v in out.sub_metrics.items()}  # drop det proposal
                # RMS-of-difference is symmetric, so the same helper serves both frames.
                ec = two_frame_extended_comfort(out.simulated_states[1:], det_states[partner], dt)
                base = epdms_score(roll_sub, ec=ec)
                rewards[i] = self._scene_bonus(base, roll_sub["ego_progress"], ec, self._gate(roll_sub))

        # -- simscale (v1 PDMS, no EC available)
        for i in range(n_nav, B):
            sub = self._score_simscale(tokens[i], trajs_np[i])
            rewards[i] = self._scene_bonus(sub["score"], sub["ego_progress"], None, sub["gate"])

        # -- group-standardized advantages with degenerate-group filtering
        rewards_t = torch.as_tensor(rewards, device=trajs.device, dtype=torch.float32)
        mean_r = rewards_t.mean(dim=1, keepdim=True)
        std_r = rewards_t.std(dim=1, keepdim=True)
        active = (std_r > self._degenerate_std_threshold).float()
        advantages = ((rewards_t - mean_r) / std_r.clamp(min=self._degenerate_std_threshold)) * active
        active_frac = active.mean()

        num_denoising_steps = chains.shape[1] - 1
        idx = torch.arange(num_denoising_steps, device=trajs.device)
        discount = planner.gamma_denoising ** (num_denoising_steps - idx - 1)
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
        adv_weighted_flat = (adv_steps * discount.view(1, 1, -1).expand(B, G, -1)).reshape(-1)

        log_probs = planner.get_logprobs(vl_rep, his_rep, status_rep, chains, deterministic=False)
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])

        n_active_terms = (active.expand(-1, G).reshape(-1, 1) * torch.ones(1, num_denoising_steps, device=trajs.device)).sum()
        policy_loss = -(log_probs * adv_weighted_flat).sum() / torch.clamp(n_active_terms, min=1.0)
        total_loss = policy_loss

        bc_loss = torch.tensor(0.0, device=trajs.device)
        if self._bc_coeff > 0:
            with torch.no_grad():
                teacher_chains, _ = planner.old_policy.sample_chain(
                    vl_features, his_traj, status_feature, deterministic=False
                )
            bc_logp = planner.get_logprobs(vl_features, his_traj, status_feature, teacher_chains, deterministic=False)
            bc_logp = bc_logp.clamp(min=-5, max=2)
            k_steps = teacher_chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, k_steps, teacher_chains.shape[2], teacher_chains.shape[3]).mean(dim=[1, 2, 3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + self._bc_coeff * bc_loss

        # diagnostics
        nav_reward = float(rewards[:n_nav].mean()) if n_nav else float("nan")
        sim_reward = float(rewards[n_nav:].mean()) if n_nav < B else float("nan")
        ec_values = []
        for p in range(num_pairs):
            for i, partner in ((2 * p + 1, 2 * p), (2 * p, 2 * p + 1)):
                ec_values.append(
                    two_frame_extended_comfort(det_states[i][None], det_states[partner], float(pair_dt[p]))[0]
                )
        det_ec = float(np.mean(ec_values)) if ec_values else float("nan")

        return BatchFeature(
            data={
                "loss": total_loss,
                "reward": rewards_t.mean(),
                "policy_loss": policy_loss,
                "bc_loss": bc_loss,
                "reward_navtrain": torch.tensor(nav_reward),
                "reward_simscale": torch.tensor(sim_reward),
                "det_ec": torch.tensor(det_ec),
                "active_group_frac": active_frac,
            }
        )

    @staticmethod
    def _gate(sub: Dict[str, np.ndarray]) -> np.ndarray:
        return sub["no_at_fault_collisions"] * sub["drivable_area_compliance"]
