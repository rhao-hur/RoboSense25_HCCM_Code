# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from mmengine.dataset import BaseDataset
from mmengine.fileio import join_path, list_from_file, load
from mmengine import fileio
import os
from os import PathLike

import json
import os.path as osp
from collections import OrderedDict
from os import PathLike
from typing import List, Sequence, Union
import numpy as np
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS, TRANSFORMS

def expanduser(data_prefix):
    if isinstance(data_prefix, (str, PathLike)):
        return osp.expanduser(data_prefix)
    else:
        return data_prefix
    
@DATASETS.register_module()
class GeoText1652Dataset(BaseDataset):
    def __init__(self,
                ann_file: str,
                test_mode: bool = False,
                data_prefix: Union[str, dict] = '',
                data_root: str = '',
                pipeline: Sequence = (),
                **kwargs):

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            pipeline=transforms,
            ann_file=ann_file,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load and parse annotations from an annotation file.

        This method handles loading the annotations and parsing the data into
        the expected format.

        Returns:
            list[dict]: A list of parsed annotations.
        """

        (prefix_key, img_prefix),  = self.data_prefix.items()

        anno_info = json.load(open(self.ann_file, 'r'))

        building_ids = {anno['image_id'][:4] for anno in anno_info}
        building_to_idx = {bid: idx for idx, bid in enumerate(building_ids)}

        processed_images = {}
        for anno in anno_info:
            image_id = anno['image_id']
            if image_id not in processed_images:
                processed_images[image_id] = anno

        img_dict = {
            image_id: {
                'ori_id': image_id,
                'building_id': building_to_idx[image_id[:4]],
                'image_id': idx,
                'img_path': join_path(img_prefix, anno['image']),
            }
            for idx, (image_id, anno) in enumerate(processed_images.items())
        }
        if self.test_mode:
            test_list = []
            text_id_counter  = 0
            for idx, anno in enumerate(anno_info):
                img_info = img_dict[anno['image_id']]
                text_ids = list(range(text_id_counter, text_id_counter + len(anno['caption'])))
                text_id_counter += len(anno['caption'])

                test_list.append({
                    'text': anno['caption'],
                    'text_id': text_ids,
                    'img_path': img_info['img_path'],
                    'image_id': img_info['image_id'],
                    'image_ori_id': img_info['ori_id'],
                    'text2building_id': [img_info['building_id']] * len(anno['caption']),
                    'image2building_id': img_info['building_id']
                })
            
            all_text_building_id = np.concatenate([data['text2building_id'] for data in test_list])
            all_image_building_id = np.array([data['image2building_id'] for data in test_list])

            for data in test_list:
                data['gt_text_id'] = list(np.where(all_text_building_id == data['image2building_id'])[0])
                data['gt_image_id'] = [
                    list(np.where(all_image_building_id == building_id)[0])
                    for building_id in data['text2building_id']
                ]
            self.img_size = len(test_list)
            self.text_size = text_id_counter
            return test_list
        else:
            train_list = [
                {
                    'text': anno['caption'],
                    'text_id': idx,
                    'img_path': img_dict[anno['image_id']]['img_path'],
                    'image_id': img_dict[anno['image_id']]['image_id'],
                    'building_id': img_dict[anno['image_id']]['building_id'],
                    'sentences': anno['sentences'],
                    'bboxes': anno['bboxes'],
                    'entity_bboxes': anno.get('entity_bboxes', None),
                    'is_matched': True,
                    'img_pseudoLabel_path': join_path(self.data_root, "img_seg_labels", f"{idx}.npy"),
                    'txt_pseudoLabel_path': join_path(self.data_root, "txt_seg_labels", f"{idx}.npy"),
                }
                for idx, anno in enumerate(anno_info)
            ]
            self.img_size = len(train_list)
            self.text_size = len(train_list)
            return train_list
