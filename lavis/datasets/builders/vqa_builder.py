"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, COCOVQADataset_flan, COCOVQAnoriDataset_flan
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset, VGVQADataset_flan, VGVQAnoriDataset_flan
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }

@registry.register_builder("coco_vqa_flan")
class COCOVQABuilder_flan(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_flan
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa_flan.yaml",
        "eval": "configs/datasets/coco/eval_vqa_flan.yaml",
    }

@registry.register_builder("coco_vqa_flan_nori")
class COCOVQAnoriBuilder_flan(BaseDatasetBuilder):
    train_dataset_cls = COCOVQAnoriDataset_flan
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa_flan_nori.yaml",
        "eval": "configs/datasets/coco/eval_vqa_flan_nori.yaml",
    }


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}

@registry.register_builder("vg_vqa_flan")
class VGVQABuilder_flan(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset_flan
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa_flan.yaml"}

@registry.register_builder("vg_vqa_flan_nori")
class VGVQABuilder_flan_nori(BaseDatasetBuilder):
    train_dataset_cls = VGVQAnoriDataset_flan
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa_flan_nori.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }