python -m torch.distributed.run --nnodes=5 --node_rank=0 --nproc_per_node=8 --master_addr='100.125.195.254' \
train.py \
--cfg-path /home/lijun07/code/LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml