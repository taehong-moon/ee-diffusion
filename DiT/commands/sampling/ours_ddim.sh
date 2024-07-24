# Ours sampling
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3

list1="1025000"
# list2="10000"
# list3="180000 200000 220000 240000 260000 280000 300000"

for var in $list1
do
    SAMPLING_FLAGS="--is_sampling \
                    --sampler ddim \
                    --drop_schedule p1 \
                    --num-sampling-steps 100 \
                    --ckpt /home/jovyan/taehong/DiT/results/dit_imagenet_256_p1_sp1_rescale/checkpoints/$var.pt \
                    --sample-dir /home/jovyan/taehong/DiT/samples/ours/dit_imagenet_256_p1_sp1_rescale/$var/ddim/100 \
                    --num-fid-samples 5000 \
                    --vae mse \
                    --cfg-scale 1.5"
    python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 sample_ddp.py $SAMPLING_FLAGS
done

# for var in $list2
# do
#     SAMPLING_FLAGS="--is_sampling \
#                     --sampler ddim \
#                     --drop_schedule a4 \
#                     --num-sampling-steps 50 \
#                     --ckpt /home/jovyan/taehong/DiT/results/dit_imagenet_256_a4_s41/checkpoints/00$var.pt \
#                     --sample-dir /home/jovyan/taehong/DiT/samples/ours/dit_imagenet_256_a4_s41/$var/ddim/50 \
#                     --num-fid-samples 5000 \
#                     --vae mse \
#                     --cfg-scale 1.5"
#     python -m torch.distributed.launch --nproc_per_node 4 --master_port 4321 sample_ddp.py $SAMPLING_FLAGS
# done

# for var in $list3
# do
#     SAMPLING_FLAGS="--is_sampling \
#                     --sampler ddim \
#                     --drop_schedule a1 \
#                     --num-sampling-steps 50 \
#                     --ckpt /home/jovyan/taehong/DiT/results/dit_imagenet_256_a1_s1_rescale/checkpoints/0$var.pt \
#                     --sample-dir /home/jovyan/taehong/DiT/samples/ours/dit_imagenet_256_a1_s1_rescale/$var/ddim/50 \
#                     --num-fid-samples 5000 \
#                     --vae mse \
#                     --cfg-scale 1.5"
#     python -m torch.distributed.launch --nproc_per_node 4 --master_port 4321 sample_ddp.py $SAMPLING_FLAGS
# done