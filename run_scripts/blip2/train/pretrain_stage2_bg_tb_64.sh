TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.run --nproc_per_node=8 \
train.py \
--cfg-path /home/lijun07/code/LAVIS/lavis/projects/blip2/train/pretrain_stage2_bl_tb.yaml