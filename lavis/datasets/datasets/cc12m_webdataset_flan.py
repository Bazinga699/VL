"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import webdataset as wds
from lavis.datasets.datasets.base_dataset import BaseDataset
import logging
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import refile
import numpy as np


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def oss_url_opener(data, handler=log_and_continue, **kw):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = refile.smart_open(url, **kw)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = oss_url_opener(src, handler=handler, mode="rb")
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples



class CC12MwebDataset_flan(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            tarfile_to_samples_nothrow,
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.rename(image="jpg;png;jpeg", text="txt"),
            wds.to_tuple("image", "text", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
        self.inner_dataset.with_length(12423374)

    def to_dict(self, sample):
        text = self.text_processor(sample[1]).split(' ')
        split_pos = np.random.randint(0, len(text) // 2 + 1)
        text_input = ' '.join(text[:split_pos])
        text_output = ' '.join(text[split_pos:])
        
        return {
            "image": sample[0],
            "text_input": text_input,
            "text_output": text_output,
        }



if __name__ == "__main__":
    from torchvision import transforms

    def to_image_text_pair(sample):
        return sample[0], sample[1]["caption"]

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CC12MwebDataset_flan(
        vis_processor=transform_train,
        text_processor=lambda x: x,
        location='s3://research-model-hh-b/Dataset/CC12M/{00000..01099}.tar',
    )

    import torch

    loader = torch.utils.data.DataLoader(dataset.inner_dataset, batch_size=2)

    print(next(iter(loader))["image"].shape, len(loader))


