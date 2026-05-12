from typing import List, Optional, Tuple, Union
import contextlib
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig

from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _autocast_ctx(self, model_dtype: torch.dtype):
        """Align vision and text branches under AMP (fixes BF16 input_embeds vs FP32 vit)."""
        if model_dtype == torch.bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if model_dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    def _build_model_inputs(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        max_length: int = 2800,
    ):
        """Shared tokenization logic for forward and forward_with_logits."""
        model_dtype = next(self.model.parameters()).dtype
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            queries, return_tensors='pt', padding='max_length', max_length=max_length
        )
        device = pixel_values.device
        input_ids      = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        position_ids   = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        num_patches_total = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches_total, dtype=torch.long, device=device)
        return input_ids, attention_mask, position_ids, image_flags, model_dtype

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized.")

        input_ids, attention_mask, position_ids, image_flags, model_dtype = \
            self._build_model_inputs(pixel_values, questions, num_patches_list)

        with self._autocast_ctx(model_dtype):
            return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
            )

    def forward_with_logits(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        generated_input_ids: Optional[torch.Tensor] = None,
        generated_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass that returns both hidden states and logits.

        If generated_input_ids is provided (student's on-policy token sequence),
        the model runs a teacher-forcing forward on those tokens and returns logits.
        This is used for OPD: both teacher and student call this with the same
        generated_input_ids to get their respective logits for KL computation.

        Returns:
            output: model output with .logits (B, T, V), .hidden_states, .attentions
            response_mask: (B, T) bool mask marking the response tokens (not prompt/image)
        """
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized.")

        input_ids, attention_mask, position_ids, image_flags, model_dtype = \
            self._build_model_inputs(pixel_values, questions, num_patches_list)

        if generated_input_ids is not None:
            # Teacher-forcing: run forward on the student's generated sequence.
            # Concatenate prompt input_ids with generated response ids.
            full_ids  = torch.cat([input_ids,  generated_input_ids],  dim=1)
            gen_mask  = torch.ones_like(generated_input_ids)
            full_mask = torch.cat([attention_mask, gen_mask], dim=1)
            full_pos  = full_mask.long().cumsum(-1) - 1
            full_pos.masked_fill_(full_mask == 0, 1)

            with self._autocast_ctx(model_dtype):
                output = self.model(
                    pixel_values=pixel_values.to(model_dtype),
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    position_ids=full_pos,
                    image_flags=image_flags.squeeze(-1),
                    output_hidden_states=True,
                    return_dict=True,
                )
            # response_mask: generated tokens only, excluding PAD and EOS positions
            T_gen = generated_input_ids.size(1)
            special_ids: set = set()
            if self.tokenizer.pad_token_id is not None:
                special_ids.add(self.tokenizer.pad_token_id)
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                if isinstance(eos, (list, tuple)):
                    special_ids.update(eos)
                else:
                    special_ids.add(eos)
            special_token_mask = torch.ones_like(generated_input_ids, dtype=torch.bool)
            for sid in special_ids:
                special_token_mask &= (generated_input_ids != sid)
            response_mask = gen_mask.bool() & special_token_mask           # (B, T_gen)
            # logits aligned to generated tokens: shift by 1 (predict next token)
            prompt_len = input_ids.size(1)
            logits = output.logits[:, prompt_len - 1 : prompt_len - 1 + T_gen, :]
        else:
            # Standard forward without generation (returns full sequence logits)
            with self._autocast_ctx(model_dtype):
                output = self.model(
                    pixel_values=pixel_values.to(model_dtype),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags.squeeze(-1),
                    output_hidden_states=True,
                    return_dict=True,
                )
            logits = output.logits
            response_mask = attention_mask.bool()

        return output, logits, response_mask

    
