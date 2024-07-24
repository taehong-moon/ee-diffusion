# INFO
CUDA_VISIBLE_DEVICES=0,1

# Default values
EPOCH=$1
DROP_SCHEDULE=$2
LAMBDA_SCHEDULE=$3
IDENTIFIER=$4

torchrun --nproc_per_node=2 --master_port=1234 train_ase.py \
    --epoch=${EPOCH} \
    --drop_schedule=${DROP_SCHEDULE} \
    --lambda_schedule=${LAMBDA_SCHEDULE} \
    --identifier=${IDENTIFIER} \
    --full \
    --freeze_time_embedder \
    --ckpt pretrained/DiT-XL-2-256x256.pt \
    --model DiT-XL/2