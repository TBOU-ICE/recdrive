"""Optional two-process CPU DDP graph smoke test with a tiny surrogate model.
  torchrun --standalone --nproc_per_node=2 tests/smoke_correction_ddp.py
This does not validate a real ReCogDrive checkpoint or GPU dependencies.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_student_state_correction import TinyPlanner, sample
from navsim.agents.recogdrive.student_state_correction import CorrectionObjective, OPDObjective

dist.init_process_group("gloo")
for stage in ("finetune", "opd"):
    torch.manual_seed(42)
    p=TinyPlanner()
    if stage == "finetune":
        objective=CorrectionObjective(p,{"correction_weight":1.,"retention_weight":1.})
    else:
        teacher=TinyPlanner()
        objective=OPDObjective(p,teacher,{"kd_weight":1.,"loss_min_sigma":.04,"precision_clip":25.},
                               {"min_sigma":.04,"noise_clip":5.,"deterministic":False})
    model=DDP(objective,find_unused_parameters=True)
    optimizer=torch.optim.AdamW(p.parameters(),lr=1e-3)
    for step in range(2):
        torch.manual_seed(100+int(os.getenv("RANK"))+step)
        optimizer.zero_grad()
        loss,_=model([sample()],diagnostic=True)
        loss.backward()
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    if dist.get_rank()==0:
        print(stage+": two-rank forward/backward passed",flush=True)
dist.destroy_process_group()
