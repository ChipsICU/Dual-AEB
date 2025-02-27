from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})
    version: Optional[str] = field(default="v0")

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='spatial_unpad')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="one_token")


@dataclass
class DataArguments:
    is_multimodal: bool = True
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "anyres"
    image_grid_pinpoints: Optional[str] = field(default="[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]")
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=32)

    # =============== AEB Data =============== #
    train_data_path: str = field(default=None,metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None,metadata={"help": "Path to the evaluation data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    # =============== General Settings =============== #
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    
    # =============== FM =============== #
    model_max_length: int = field(
        default=32768,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # =============== Grouping =============== #
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=True, metadata={"help": "Group the input data by modality length."})
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)

    # =============== Quantization =============== #
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)

    # =============== Training =============== #
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=12)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    dataloader_num_workers: int = field(default=8)
    lr_scheduler_type: str = field(default="cosine")

    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = 2e-6
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=2)

    dataloader_drop_last: bool = field(default=False)
    # =============== Evaluation =============== #
    evaluation_strategy: str = field(default="epoch") # "no, epoch, steps" is provided

    # =============== Log =============== #
    save_strategy: str = field(default="epoch")
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=10)
    logging_steps: int = field(default=1)
    report_to: str = field(default="wandb")
    verbose_logging: bool = field(default=False)


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")
