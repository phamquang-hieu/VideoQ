from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
import tarfile
from .pipeline import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 repeat = 1,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False,):
        super().__init__()
        self.use_tar_format = True if ".tar" in data_prefix else False
        data_prefix = data_prefix.replace(".tar", "")
        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = data_prefix
        # self.data_prefix = osp.realpath(
        #     data_prefix) if data_prefix is not None and osp.isdir(
        #         data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        # video_infos: a dict of keys as video id and values under the form of dictionary containing info about individual video 
        # or a list of dict containing info about individual videos
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos
    
    def load_json_annotations_2(self):
        """load json annotations from extended ucf crime"""
        video_infos = mmcv.load(self.ann_file)
        # video_infos = {"<class_name>/filename>":[{"start": val_1, "end": val_1'}, {...}, ...]}
        results = []
        for vid, values in video_infos.items():
            results.append(dict(filename=os.path.join(self.data_prefix, vid) if self.data_prefix is not None else vid,
                                label=int(values['label']),
                                annotations=values["annotations"],
                                tar=self.use_tar_format))
        return results
    
    def load_json_annotations_3(self):
        video_infos = mmcv.load(self.ann_file)
        results = []
        for vid in video_infos:
            vid_name = vid['filename']
            results.append(dict(filename=os.path.join(self.data_prefix, vid_name) if self.data_prefix is not None else vid_name,
                                label=int(vid['label']),
                                annotations=vid['annotations'],
                                tar=self.use_tar_format
                                )
                        )
        return results


    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        aug1 = self.pipeline(results)
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {"imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                    "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)

class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, normal_label=13, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file
        self.normal_indices = []
        self.abnormal_indices = []
        for i, video in enumerate(self.video_infos):
            if video['label'] == normal_label:
                self.normal_indices.append(i)
            else:
                self.abnormal_indices.append(i)
        self.normal_indices = np.array(self.normal_indices)
        self.abnormal_indices = np.array(self.abnormal_indices)

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations_3()
            # return self.load_json_annotations_2()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class: # if a video belongs to multiple classes
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split 
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        return video_infos

class BalanceBatchSampler(torch.utils.data.Sampler):
    """Balancing the number of positive and negative samples within a batch"""

    def __init__(self, normal_indices, abnormal_indices, batch_size=6) -> None:
        self.normal_indices = normal_indices #[#data points]
        self.abnormal_indices = abnormal_indices #[#data points]
        self.batch_size = batch_size
        self.num_normal = len(normal_indices)
        self.num_abnormal = len(abnormal_indices)
    
    def random_sample_extend(self, array:np.ndarray, desired_len):
        return np.concatenate((array, np.random.choice(array, size=desired_len-len(array), replace=False)), axis=-1)
    
    def truncate_indices(self, array, half):
        end = len(array)%half
        if end == 0: return array.reshape(-1, half)
        return array[:-end].reshape(-1, half) 

    def __iter__(self):
        half = self.batch_size//2
        normal_indices = np.random.permutation(self.normal_indices)
        abnormal_indices = np.random.permutation(self.abnormal_indices)        
        if self.num_normal < self.num_abnormal:
            normal_indices = self.random_sample_extend(normal_indices, self.num_abnormal)
        elif self.num_normal > self.num_abnormal:
            abnormal_indices = self.random_sample_extend(abnormal_indices, self.num_normal)
        
        normal_indices = self.truncate_indices(normal_indices, half=half)
        abnormal_indices = self.truncate_indices(abnormal_indices, half=half)

        batch = np.concatenate((abnormal_indices, normal_indices), axis=-1)
        return (batch[i] for i in range(len(batch)))
    
    def __len__(self):
        """This function returns the number of batch"""
        return int(self.num_normal//(0.5*self.batch_size)) if self.num_normal > self.num_abnormal else int(self.num_abnormal//(self.batch_size*0.5))
class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleAnnotatedFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)), # assign np.inf to long edge for rescaling short edge later.
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
        
    
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # sampler_train = BalanceBatchSampler(normal_indices=train_data.normal_indices, abnormal_indices=train_data.abnormal_indices, batch_size=config.TRAIN.BATCH_SIZE) #@TODO: finish this line
    # train_loader = DataLoader(
    #     train_data, batch_sampler=sampler_train,
    #     num_workers=1,
    #     pin_memory=True,
    #     collate_fn=partial(mmcv_collate, samples_per_gpu=config.TEST.BATCH_SIZE)
    # )
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleAnnotatedFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        # config.TEST.NUM_CLIP -> number of clips sampled out of a video
        val_pipeline[1] = dict(type='SampleAnnotatedFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline)
    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size()) # assume having 4 processes -> process #0 handles indices 0, 4, 8, 12,... process #2 handles indices 1, 5, 9, 13,...
    sampler_val = SubsetRandomSampler(indices)
    
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=config.TEST.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TEST.BATCH_SIZE),
    )

    return train_data, val_data, train_loader, val_loader