# Ours sampling
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1

var="?"

SAMPLING_FLAGS="--is_sampling \
                --sampler ddim \
                --drop_schedule p1 \
                --num-sampling-steps 100 \
                --ckpt checkpoints/$var.pt \
                --sample-dir samples/$var/ddim/100 \
                --num-fid-samples 5000 \
                --vae mse \
                --cfg-scale 1.5"
torchrun --nproc_per_node=2 --master_port=1234 sample_ddp.py $SAMPLING_FLAGS