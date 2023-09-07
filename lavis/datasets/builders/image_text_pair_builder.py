"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairNoriDataset_flan, ImageTextPairNoriDataset,ImageTextPairNoriDataset_flan_prompt
from lavis.datasets.datasets.laion_dataset import LaionDataset
from lavis.datasets.datasets.cc12m_webdataset import CC12MwebDataset
from lavis.datasets.datasets.cc12m_webdataset_flan import CC12MwebDataset_flan



@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m.yaml"
    }

@registry.register_builder("conceptual_caption_12m_web")
class ConceptualCaption12MwebBuilder(BaseDatasetBuilder):
    train_dataset_cls = CC12MwebDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/conceptual_caption/defaults_12m_web.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

@registry.register_builder("conceptual_caption_12m_web_flan")
class ConceptualCaption12MwebBuilder_flan(BaseDatasetBuilder):
    train_dataset_cls = CC12MwebDataset_flan

    DATASET_CONFIG_DICT = {"default": "configs/datasets/conceptual_caption/defaults_12m_web_flan.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}

@registry.register_builder("vg_caption_flan_nori")
class VGCaptionBuilder_flan_nori(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset_flan

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_flan_nori.yaml"}

@registry.register_builder("vg_caption_flan_nori_prompt")
class VGCaptionBuilder_flan_nori(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairNoriDataset_flan_prompt

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_flan_nori_prompt.yaml"}



@registry.register_builder("laion2B_multi")
class Laion2BMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_2B_multi.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

