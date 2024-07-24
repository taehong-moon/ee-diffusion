# Baseline sampling
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1

steps="200 100 50"
for var in $steps
do
    SAMPLING_FLAGS="--is_sampling \
                    --baseline \
                    --sampler ddim \
                    --num-sampling-steps $var \
                    --ckpt pretrained/DiT-XL-2-256x256.pt \
                    --sample-dir samples/baseline/ddim/$var \
                    --num-fid-samples 100 \
                    --vae mse \
                    --cfg-scale 1.5"

    torchrun --nproc_per_node=2 --master_port=1234 sample_ddp.py $SAMPLING_FLAGS
done