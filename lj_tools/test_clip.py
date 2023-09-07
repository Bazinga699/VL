import torch
import clip

model, preprocess = clip.load("ViT-B/32", device='cuda')
dummy_input = torch.rand(2,3,224,224).cuda().half()
out = model.visual(dummy_input)
a = 1
