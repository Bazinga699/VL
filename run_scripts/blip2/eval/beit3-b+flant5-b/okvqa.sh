TRANSFORMERS_OFFLINE=1 \
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path /home/lijun07/code/LAVIS/lavis/projects/blip2/eval/beit3-b+flant5-b/okvqa.yaml