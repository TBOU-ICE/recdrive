# OPD (On-Policy Distillation) trainer — Teacher-TopK Local Support Matching
# arXiv:2603.25562 §3.2  L_LSM = KL(π̂_student || q̂_teacher) on top-K teacher support
# No reward weighting; pure distillation loss averaged over valid response tokens.

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


def compute_topk_kl_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    teacher_logits: torch.Tensor,   # (B, T, V)
    response_mask: torch.Tensor,    # (B, T)  1 = valid response token
    topk: int = 32,
) -> torch.Tensor:
    """
    Teacher-TopK Local Support Matching (LSM) loss.

    At each token position t, restrict both distributions to the teacher's
    top-K vocabulary support S = TopK_q(c_t), renormalize both within that
    support via softmax, then compute KL(π̂_student || q̂_teacher).

    Returns scalar loss averaged over valid (non-PAD, non-EOS) response tokens.
    """
    # teacher top-K indices at every position
    _, ref_topk_idx = teacher_logits.topk(topk, dim=-1)           # (B, T, K)

    # gather logits restricted to teacher support
    ref_logits_k   = teacher_logits.gather(-1, ref_topk_idx)       # (B, T, K)
    actor_logits_k = student_logits.gather(-1, ref_topk_idx)       # (B, T, K)

    # renormalize within support → π̂ and q̂
    teacher_log_norm  = F.log_softmax(ref_logits_k,   dim=-1)      # log q̂
    student_log_norm  = F.log_softmax(actor_logits_k, dim=-1)      # log π̂
    student_prob_norm = student_log_norm.exp()                      # π̂

    # KL(π̂ || q̂) per token
    kl_per_token = (student_prob_norm * (student_log_norm - teacher_log_norm)).sum(dim=-1)

    # mask out padding / EOS and average
    kl_per_token = kl_per_token * response_mask
    n_valid = response_mask.sum().clamp(min=1)
    return kl_per_token.sum() / n_valid


class ReCogDriveOPDTrainer:
    """Computes Teacher-TopK LSM distillation loss (paper: arXiv:2603.25562)."""

    def __init__(self, topk: int = 32):
        self.topk = topk

    def compute_loss(
        self,
        student_logits: torch.Tensor,   # (B*G, T, V)
        teacher_logits: torch.Tensor,   # (B*G, T, V)
        response_mask: torch.Tensor,    # (B*G, T)  float
    ) -> BatchFeature:
        opd_loss = compute_topk_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            response_mask=response_mask,
            topk=self.topk,
        )
        opd_loss = torch.nan_to_num(opd_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return BatchFeature(data={
            "loss":     opd_loss,
            "opd_loss": opd_loss.detach(),
        })
