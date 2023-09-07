from lavis.tasks.captioning import coco_caption_eval
import os
from lavis.common.registry import registry


results_file = '/data/BLIP2/coco_caption_eval/20230321151/result/test_epochbest.json'
coco_caption_eval(os.path.join(registry.get_path("cache_root"), "coco_gt"), results_file, 'test')
