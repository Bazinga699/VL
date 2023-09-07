from transformers import BertTokenizer
import os
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
t5_model = 'google/flan-t5-small'

t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
t5_config = T5Config.from_pretrained(t5_model)
t5_config.dense_act_fn = "gelu"
t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
# os.environ['TRANSFORMERS_OFFLINE']="1"

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.save_pretrained('/home/lijun07/.cache/huggingface/hub/models--bert-base-uncased/vocab')

# from transformers import T5TokenizerFast
# t5_tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-base')
# tokenizer.save_pretrained('/home/lijun07/.cache/huggingface/hub/models--bert-base-uncased/')
a = 1