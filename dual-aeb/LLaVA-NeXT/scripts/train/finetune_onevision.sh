export WANDB_PROJECT="dual-aeb"
export TOKENIZERS_PARALLELISM=True
export CUDA_VISIBLE_DEVICES=
MID_RUN_NAME="b2d_openloop_brake_only_meanpooling_image"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"
MODEL_PATH="./pretrain_ckpt/llava-onevision-qwen2-0.5b-ov" 
VISION_MODEL="./pretrain_ckpt/siglip-so400m-patch14-384"
TRAIN_DATA_PATH=
EVAL_DATA_PATH=
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03

############### Pretrain ################
export NUM_GPUS=6          
export NNODES=1            
export RANK=0              
export ADDR="localhost"    # single node is localhost
export PORT=12345          # master port

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL} \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --model_max_length 32768 \
    --torch_compile True \
    --torch_compile_backend "inductor" \

# You can delete the sdpa attn_implementation if you want to use flash attn