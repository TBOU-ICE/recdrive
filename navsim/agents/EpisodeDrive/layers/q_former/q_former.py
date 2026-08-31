from pathlib import Path
import importlib.util
import sys

_src = Path('/mnt/volumes/ad-e2e-al-sh01/nby/HUGSIM_DriveMem_base/EpisodeDrive/layers/q_former/q_former.py')
_spec = importlib.util.spec_from_file_location('_episode_drive_qformer', _src)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
VisionOnlyQFormer = _mod.VisionOnlyQFormer
