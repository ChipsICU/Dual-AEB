from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print

# only provide qwen
def load_pretrained_model(model_path, model_name, load_8bit=False, load_4bit=False, device_map="auto", attn_implementation="flash_attention_2", overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float16

    rank0_print(f"Loaded LLaVA model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "qwen" in model_name.lower() or "quyen" in model_name.lower():
        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
        if overwrite_config is not None:
            rank0_print(f"Overwriting config with {overwrite_config}")
            for k, v in overwrite_config.items():
                setattr(llava_cfg, k, v)
            model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
        else:
            model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

    else:
        raise ValueError(f"Model {model_name} not supported")

    # Image processor related setup (if any)
    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:  
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
