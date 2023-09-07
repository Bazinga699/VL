import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

raw_image = Image.open("/home/lijun07/code/LAVIS/docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the sea"

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
sample = {"image": image, "text_input": [text_input]}