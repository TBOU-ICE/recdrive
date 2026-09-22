"""CPU numerical/contract tests; no real checkpoints, VLM or NAVSIM maps required.
Run after overlay: python -m unittest discover -s tests -p test_student_state_correction.py -v
"""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from navsim.agents.recogdrive.student_state_correction import (
    CorrectionObjective, OPDObjective, assert_disjoint, encode, errors, inputs,
    planner_state, read_indexes, rollout, seed_for, summary, teacher_prediction,
)
from navsim.agents.recogdrive.recogdrive_goalfree_kd_only_distill_trainer import ReCogDriveGoalFreeKDOnlyDistillTrainer as KD


class TinyPlanner(nn.Module):
    """Differentiable DDIM kernel for tests, NOT a ReCogDrive performance proxy."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=8, action_dim=3, sampling_method="ddim")
        self.feature_encoder = nn.Linear(3, 3)
        self.his_traj_encoder = nn.Linear(12, 3)
        self.ego_status_encoder = nn.Linear(8, 3)
        self.action_decoder = nn.Linear(3, 3)
        nn.init.zeros_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)
        self.ddim_steps = 5
        self.register_buffer("ddim_t", torch.tensor([80, 60, 40, 20, 0]))
        self.register_buffer("ddim_alphas", torch.tensor([.2, .4, .6, .8, .99]))
        self.register_buffer("ddim_alphas_prev", torch.tensor([.4, .6, .8, .99, 1.]))
        self.register_buffer("ddim_sqrt_one_minus_alphas", (1-self.ddim_alphas).sqrt())
        self.goal_dropout_p = .1
        self.goal_noise_p = .2
        self.denoised_clip_value = 1.
        self.goal = None

    @contextlib.contextmanager
    def goal_context(self, goal):
        old, self.goal = self.goal, goal
        try:
            yield
        finally:
            self.goal = old

    def _goal_embedding_for(self, goal, dtype):
        return goal

    def _dit_step(self, z, t, vl, his, ego, goal):
        return self.action_decoder(z) + .001 * (vl.mean() + his.mean() + ego.mean())

    @staticmethod
    def extract(a, idx, shape):
        return a[idx].reshape(-1, 1, 1)

    @staticmethod
    def make_timesteps(b, t, device):
        return torch.full((b,), int(t), dtype=torch.long, device=device)

    @staticmethod
    def norm_odo(x):
        return x / 10

    @staticmethod
    def denorm_odo(x):
        return x * 10

    def p_mean_variance(self, z, t, idx, vl, his, ego, deterministic=False):
        eps = self._dit_step(z, t, vl, his, ego, self.goal)
        alpha = self.extract(self.ddim_alphas, idx, z.shape)
        prev = self.extract(self.ddim_alphas_prev, idx, z.shape)
        x0 = ((z - (1-alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1, 1)
        sigma = ((1-prev)/(1-alpha)*(1-alpha/prev)).sqrt()
        if deterministic:
            sigma = torch.zeros_like(sigma)
        mean = KD._shared_ddim_mean(self, z, idx, x0, sigma)
        return mean, torch.log(sigma.square() + 1e-20), x0

    def forward(self, vl, action):
        enc = encode(self, vl, action)
        target = torch.randn_like(action.action)
        eps = self._dit_step(target, None, *enc, action.goal)
        return {"loss": (eps-target).square().mean()}


def sample():
    return {"token": "abc", "path": "/cache/log_a/abc", "seed": 1,
            "vl": torch.ones(2, 3), "history": torch.ones(12), "status": torch.ones(8),
            "gt": torch.ones(8, 3), "student_final": torch.ones(8, 3),
            "states": [{"step": 4, "timestep": 0, "z": torch.full((8, 3), .2)}]}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.p = TinyPlanner()
        self.vl, self.action = inputs(sample(), "cpu")

    def test_checkpoint_selects_only_student(self):
        state = {"agent.action_head.action_decoder.weight": torch.ones(1),
                 "agent.teacher_planners.single.action_decoder.weight": torch.zeros(1)}
        self.assertEqual(list(planner_state({"state_dict": state})), ["action_decoder.weight"])

    def test_overlapping_logs_fail(self):
        with self.assertRaisesRegex(ValueError, "log overlap"):
            assert_disjoint([{"token": "a", "path": "/cache/log/a"}], [{"token": "b", "path": "/cache/log/b"}])

    def test_overlapping_tokens_fail(self):
        with self.assertRaisesRegex(ValueError, "token overlap"):
            assert_disjoint([{"token": "a", "path": "/cache/log1/a"}], [{"token": "a", "path": "/cache/log2/a"}])

    def test_index_formats_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a.json", Path(tmp)/"b.json"
            a.write_text(json.dumps({"builders": ["internvl_feature", "trajectory_target"], "cache_path": "/cache", "tokens": {"a": "log/a"}}))
            b.write_text(json.dumps({"builders": ["internvl_feature", "trajectory_target"], "samples": [{"token": "a", "path": "/cache/log/a"}]}))
            self.assertEqual(len(read_indexes([a, b])), 1)
            b.write_text(b.read_text().replace("/cache/log/a", "/other/log/a"))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_indexes([a,b])

    def test_teacher_prediction_matches_kernel(self):
        enc = encode(self.p, self.vl, self.action)
        z = sample()["states"][0]["z"].unsqueeze(0)
        x0, _, _ = teacher_prediction(self.p, z, 4, enc, self.action.goal)
        _, _, expected = self.p.p_mean_variance(z, torch.tensor([0]), torch.tensor([4]), *enc)
        torch.testing.assert_close(x0, expected)

    def test_terminal_shared_mean_equals_x0(self):
        z = torch.randn(1,8,3)
        x = torch.randn_like(z)
        mu = KD._shared_ddim_mean(self.p, z, torch.tensor([4]), x, torch.zeros(1,1,1))
        torch.testing.assert_close(mu,x)

    def test_collection_matches_opd_sampler_and_pre_call_state(self):
        settings = {"min_sigma": .04, "noise_clip": 5., "deterministic": False}
        kd = KD(bucket_names=["single"], rollout_min_sigma=.04, rollout_noise_clip=5.)
        torch.manual_seed(11)
        states, _, _ = rollout(self.p, self.vl, self.action, settings)
        torch.manual_seed(11)
        enc = encode(self.p,self.vl,self.action)
        chain = kd._sample_chain(self.p, *enc, 1, torch.device("cpu"), torch.float32)
        self.assertEqual(len(states),5)
        torch.testing.assert_close(states[4], chain[:,4])
        self.assertFalse(torch.equal(states[4], chain[:,5]))
        self.assertFalse(states[4].requires_grad)

    def test_loss_floor_does_not_change_states(self):
        enc = encode(self.p,self.vl,self.action)
        chains=[]
        for floor in (.04,.2):
            kd=KD(bucket_names=["single"], min_sigma=floor, rollout_min_sigma=.0001, rollout_noise_clip=1.)
            torch.manual_seed(2)
            chains.append(kd._sample_chain(self.p,*enc,1,torch.device("cpu"),torch.float32))
        torch.testing.assert_close(*chains)

    def test_correction_gradient_finite_and_improves(self):
        obj=CorrectionObjective(self.p,{"correction_weight":1.,"retention_weight":0.})
        opt=torch.optim.SGD(self.p.parameters(),lr=10.)
        before,_=obj([sample()],True)
        initial=float(before.detach())
        for _ in range(15):
            opt.zero_grad()
            loss,rows=obj([sample()],True)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()),initial)
        self.assertGreater(rows[0]["z4/corr_grad_eps_norm"],0)
        self.assertEqual((self.p.goal_dropout_p,self.p.goal_noise_p),(.1,.2))

    def test_opd_backward_and_teacher_frozen(self):
        teacher=TinyPlanner()
        teacher.action_decoder.bias.data.fill_(.2)
        obj=OPDObjective(self.p,teacher,{"kd_weight":1.,"loss_min_sigma":.04,"precision_clip":25.},
                         {"min_sigma":.04,"noise_clip":5.,"deterministic":False})
        loss,rows=obj([sample()],True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("kd_grad_x0_step_4",rows[0])
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(self.p.action_decoder.weight.grad.abs().sum().item(),0.)

    def test_fixed_state_validation_paired_gain(self):
        from navsim.planning.script.run_student_state_correction import evaluate_teacher
        baseline=TinyPlanner()
        # Positive epsilon corrects the biased positive z toward GT.
        self.p.action_decoder.bias.data.fill_(.5)
        cfg={"sampler":{"min_sigma":.04,"noise_clip":5.,"deterministic":False}}
        report,rows=evaluate_teacher(self.p,[sample()],cfg,torch.device("cpu"),baseline,export=True)
        self.assertGreater(rows[0]["paired_ade_gain_m"],0)
        self.assertIn("teacher_x0_trajectory",rows[0])
        self.assertEqual(report["4"]["n"],1)

    def test_pipeline_collect_train_export(self):
        import gzip
        import pickle
        from navsim.planning.script import run_student_state_correction as run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            indexes=[]
            for split in ("train","val"):
                folder=root/f"log_{split}"/split
                folder.mkdir(parents=True)
                datum=sample()
                features={"last_hidden_state":datum["vl"], "history_trajectory":datum["history"], "status_feature":datum["status"]}
                for name,value in (("internvl_feature",features),("trajectory_target",{"trajectory":datum["gt"]})):
                    with gzip.open(folder/f"{name}.gz","wb") as f:
                        pickle.dump(value,f)
                index=root/f"{split}.json"
                index.write_text(json.dumps({"builders":["internvl_feature","trajectory_target"],"samples":[{"token":split,"path":str(folder)}]}))
                indexes.append(str(index))
            checkpoint=root/"input.ckpt"
            checkpoint.write_bytes(b"test provenance only; fake model factory")
            spec={"checkpoint":str(checkpoint),"kind":"goal_free"}
            cfg={"seed":3,"state_dir":str(root/"states"),"output_dir":str(root/"output"),
                 "train_indexes":[indexes[0]],"val_indexes":[indexes[1]],
                 "collector_student":spec,"teacher":dict(spec,kind="privileged"),"opd_student":spec,
                 "sampler":{"min_sigma":.04,"noise_clip":5.,"deterministic":False},
                 "collect":{"steps":[4],"seeds_per_scene":1},
                 "finetune":{"epochs":1,"batch_size":1,"workers":0,"lr":.001,"weight_decay":0.,
                             "correction_weight":1.,"retention_weight":1.,"grad_clip":1.,"diagnostic_interval":1},
                 "opd":{"epochs":1,"batch_size":1,"workers":0,"lr":.001,"weight_decay":0.,
                         "kd_weight":1.,"loss_min_sigma":.04,"precision_clip":25.,"grad_clip":1.,"diagnostic_interval":1}}
            with patch.object(run,"build_planner",side_effect=lambda *args: TinyPlanner()):
                run.collect(cfg,torch.device("cpu"),0,1)
                manifest=json.loads((root/"states/train/manifest.json").read_text())
                self.assertEqual(manifest["schema"],2)
                state=torch.load(root/"states/train"/manifest["samples"][0]["file"],weights_only=False)
                self.assertEqual(state["schema"],2)
                self.assertTrue({"vl","history","status","gt"}.isdisjoint(state))
                state_rows = [dict(manifest["samples"][0], file=str(root/"states/train"/manifest["samples"][0]["file"]))]
                hydrated=run.Samples(state_rows,states=True)[0]
                self.assertTrue({"vl","history","status","gt"}.issubset(hydrated))
                run.train(cfg,"finetune",torch.device("cpu"),0,1,"original")
                run.train(cfg,"opd",torch.device("cpu"),0,1,"original")
                run.train(cfg,"opd",torch.device("cpu"),0,1,"adapted")
            result=torch.load(root/"output/finetune/best.ckpt",weights_only=False)
            self.assertIn("action_decoder.weight",planner_state(result))
            self.assertTrue((root/"output/opd_adapted/val_epoch_000.json").exists())

    def test_pdms_scores_fixed_x0_separately_from_full_rollout(self):
        from navsim.planning.script import run_student_state_correction as run
        cfg={"metric_cache_path":"fake", "sampler":{"min_sigma":.04,"noise_clip":5.,"deterministic":False}}
        calls=[]
        def score(token,trajectory):
            calls.append(trajectory.clone())
            return {"score":.8}
        with patch.object(run,"PDMScoring",return_value=score):
            _,rows=run.evaluate_teacher(self.p,[sample()],cfg,torch.device("cpu"),TinyPlanner(),True,True,True)
        self.assertEqual(len(calls),4)
        torch.testing.assert_close(calls[0],torch.tensor(rows[0]["teacher_x0_trajectory"]))
        self.assertIn("x0_pdm_score",rows[0])
        self.assertIn("full_pdms",rows[1])

    def test_seed_and_metrics(self):
        self.assertEqual(seed_for("a",42),seed_for("a",42))
        self.assertNotEqual(seed_for("a",42),seed_for("b",42))
        self.assertAlmostEqual(errors(torch.ones(1,8,3),torch.zeros(1,8,3))["ade_m"],2**.5,places=5)
        self.assertEqual(summary([{"ade_m":1.},{"ade_m":3.}])["ade_m"]["mean"],2.)


if __name__ == "__main__":
    unittest.main()
