"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAnoriDataset

import cv2
import io
import numpy as np

class VGVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        # TODO this should be configured better
        weights = [0.2]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }

class VGVQADataset_flan(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        # TODO this should be configured better
        weights = [0.2]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "text_output": answers,
            "weights": weights,
        }

class VGVQAnoriDataset_flan(VQAnoriDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        self._check_nori_fetcher()
        ann = self.annotation[index]

        nori_id = ann["nori_id"]
        img_bytes = self.nori_fetcher.get(nori_id)
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except:
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        image = self.vis_processor(img)
        question = self.text_processor(ann["question"])

        answer = ann["answer"]
        # TODO this should be configured better
        weights = [0.2]

        return {
            "image": image,
            "text_input": question,
            "answers": answer,
            "text_output": answer,
            "weights": weights,
        }
