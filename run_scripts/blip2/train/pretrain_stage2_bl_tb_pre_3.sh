TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.run --nnodes=5 --node_rank=3 --nproc_per_node=8 --master_addr='100.122.54.172' \
train.py \
--cfg-path /home/lijun07/code/LAVIS/lavis/projects/blip2/train/pretrain_stage2_bl_tb_pre.yaml