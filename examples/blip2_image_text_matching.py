import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

caption = "merlion in Singapore"

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)