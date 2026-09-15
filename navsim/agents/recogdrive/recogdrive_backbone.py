from typing import List, Optional, Tuple, Union
import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

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
                 weight_path: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): Path to the base model (config + weights).
            weight_path (str, optional): Path to a finetuned HuggingFace checkpoint
                (must contain config.json + model.safetensors). When provided, the
                model is loaded from weight_path instead of checkpoint_path, while
                the tokenizer is still loaded from checkpoint_path.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.model_type = model_type.lower()
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from base path: '{checkpoint_path}'"
              + (f", overlaying finetuned weights from: '{weight_path}'" if weight_path else ""))

        if self.model_type == 'internvl':
            # Load architecture + custom code from base path (has configuration_internvl_chat.py etc.)
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            # Overlay finetuned weights on top of the base model
            if weight_path:
                self._load_finetuned_weights(weight_path)
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
            if weight_path:
                self._load_finetuned_weights(weight_path)
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _load_finetuned_weights(self, weight_path: str) -> None:
        """Load model.safetensors from weight_path and apply to self.model (strict=False)."""
        safetensors_file = os.path.join(weight_path, "model.safetensors")
        bin_file = os.path.join(weight_path, "pytorch_model.bin")
        if os.path.exists(safetensors_file):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_file, device="cpu")
            print(f"Loading finetuned weights from: {safetensors_file}")
        elif os.path.exists(bin_file):
            state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)
            print(f"Loading finetuned weights from: {bin_file}")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found under {weight_path}"
            )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[weight_path] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[weight_path] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        print(f"Finetuned weights loaded successfully from: {weight_path}")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
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
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding='max_length', max_length=2800)
        device = torch.device('cuda')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)


        return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    