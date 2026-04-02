# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Submodules that import flash_attn at import time are loaded lazily so that
# training without flash-attn (and without packed dataset) can start.

import importlib

from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit
from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import (concat_pad_data_collator,
                                dpo_concat_pad_data_collator,
                                pad_data_collator)
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

_LAZY_IMPORTS = {
    'replace_internlm2_attention_class': '.internlm2_packed_training_patch',
    'replace_qwen2_attention_class': '.qwen2_packed_training_patch',
    'replace_phi3_attention_class': '.phi3_packed_training_patch',
    'replace_llama_attention_class': '.llama_packed_training_patch',
    'replace_llama2_attn_with_flash_attn': '.llama2_flash_attn_monkey_patch',
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys())))


__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm',
           'replace_llama2_attn_with_flash_attn',
           'replace_train_sampler',
           'replace_train_dataloader',
           'replace_internlm2_attention_class',
           'replace_qwen2_attention_class',
           'replace_phi3_attention_class',
           'replace_llama_attention_class',
           'pad_data_collator',
           'dpo_concat_pad_data_collator',
           'concat_pad_data_collator',
           'apply_liger_kernel_to_internvit']
