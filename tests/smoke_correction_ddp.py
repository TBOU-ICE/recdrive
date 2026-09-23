"""Optional two-process CPU DDP graph smoke test with a tiny surrogate model.
  torchrun --standalone --nproc_per_node=2 tests/smoke_correction_ddp.py
This does not validate a real ReCogDrive checkpoint or GPU dependencies.
"""
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_student_state_correction import TinyPlanner, sample
from navsim.agents.recogdrive.student_state_correction import CorrectionObjective, OPDObjective
from navsim.planning.script.run_student_state_correction import validate_epoch

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
        # Branch diagnostics use autograd.grad and must not be run through the
        # DDP wrapper.  Exercise the same no_sync + regular-update pattern as
        # the production state-correction trainer.
        if dist.get_rank() == 0:
            with model.no_sync():
                _, diagnostic_rows = objective([sample()], diagnostic=True)
            assert diagnostic_rows
        loss,_=model([sample()],diagnostic=False)
        loss.backward()
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        # Exercise the epoch transition, not just training backward: the former
        # rank-0-only validation made other ranks time out at the final barrier.
        cfg = {stage: {"validation_samples": 0}, "seed": 42, "collect": {"steps": [4]},
               "sampler": {"min_sigma": .04, "noise_clip": 5., "deterministic": False}}
        for count in (5, 1):  # Uneven shards and ranks with no validation samples.
            data = [dict(sample(), token=f"val-{i}", seed=i + 1) for i in range(count)]
            rng = torch.get_rng_state().clone()
            py_rng, np_rng = random.getstate(), np.random.get_state()
            report, criterion = validate_epoch(p, data, cfg, stage, torch.device("cpu"),
                                               dist.get_rank(), dist.get_world_size(), step,
                                               chunk_size=1)
            assert torch.equal(rng, torch.get_rng_state())
            assert random.getstate() == py_rng
            assert np.array_equal(np.random.get_state()[1], np_rng[1])
            if dist.get_rank() == 0:
                expected, expected_criterion = validate_epoch(
                    p, data, cfg, stage, torch.device("cpu"), 0, 1, step, chunk_size=2)
                assert report == expected and criterion == expected_criterion
                assert (report["4"]["n"] if stage == "finetune" else report["n"]) == count
            dist.barrier()
        model.train()
        flat = torch.cat([param.detach().flatten() for param in p.parameters()])
        copies = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(copies, flat)
        for other in copies:
            torch.testing.assert_close(flat, other)
    if dist.get_rank()==0:
        print(stage+": two-rank train/validation/train passed; exact metrics and RNG preserved",flush=True)

class BrokenValidation:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 1:
            raise ValueError("intentional validation read error")
        return sample()

try:
    validate_epoch(p, BrokenValidation(), cfg, "opd", torch.device("cpu"),
                   dist.get_rank(), dist.get_world_size(), 0, chunk_size=1)
except RuntimeError as exc:
    assert "intentional validation read error" in str(exc)
else:
    raise AssertionError("validation errors must propagate to every rank")
dist.barrier()
if dist.get_rank() == 0:
    print("validation shard failure propagated on all ranks", flush=True)
dist.destroy_process_group()
