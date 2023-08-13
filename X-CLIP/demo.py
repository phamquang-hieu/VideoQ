import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import numpy as np
import random
# from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from sklearn.metrics import classification_report
import clip
import pandas as pd
from mmcv.utils import Registry
from datasets.pipeline import *

def generate_text(classes: list):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in classes])

    return classes

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config

def sum_by_index(similarity: torch.Tensor, indices: np.ndarray, n_classes=14):
    result = torch.zeros([similarity.shape[0], n_classes], device=similarity.device)
    for b_id, b in enumerate(similarity):
        for i, item in enumerate(b):
            result[b_id, indices[i]] += item
    return result

PIPELINES = Registry('pipeline')
def main(config): 
    # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

    model = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                         pool_size=config.MODEL.POOL_SIZE,
                         pool_use_freq=config.TRAIN.POOL_USE_FREQ,
                         pool_prompts_per_sample=config.MODEL.POOL_PROMPTS_PER_SAMPLE,
                         pool_prompt_length=config.MODEL.POOL_PROMPT_LENGTH,
                         pool_freeze_video=config.TRAIN.POOL_FREEZE_VIDEO,
                         num_classes=config.DATA.NUM_CLASSES,
                         context_prompt_len=config.MODEL.CONTEXT_PROMPT_LEN,
                         class_prompt_len=config.MODEL.CLASS_PROMPT_LEN,
                         fine_grain_loss=config.TRAIN.FINE_GRAIN_LOSS
                        )
    
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]

    pipeline = Compose(val_pipeline)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)
    labels = pd.read_csv(config.DATA.LABEL_LIST).values.tolist()
    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, None, None, logger)

    text_id = np.array([p[0] for p in labels])
    text_inputs = generate_text(labels).cuda()
    
    modality = 'RGB'
    start_index = 0
    video_info = dict(filename=config.DATA.SINGLE_FILE, label=-1, tar=False, modality=modality, start_index=start_index)
    
    result = copy.deepcopy(video_info)

    video = pipeline(result)['imgs'].cuda()
    video = video.unsqueeze(0)

    b, tn, c, h, w = video.size()
    t = config.DATA.NUM_FRAMES # number of frames in a video
    n = tn // t # number of views
    _image = video.view(b, n, t, c, h, w)
    
    tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
    for i in range(n): # for view in views
        image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
        image_input = image.cuda(non_blocking=True)

        # if config.TRAIN.OPT_LEVEL == 'O2':
        #     image_input = image_input.half()
        with torch.cuda.amp.autocast(enabled=True):
            output, _ = model(image_input, text_inputs)
        
        similarity = output.view(b, -1).softmax(dim=-1)
        similarity = sum_by_index(similarity, text_id, n_classes=config.DATA.NUM_CLASSES)
        tot_similarity += similarity # accumulating simmilarity from views
    
    print(similarity)


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.disabled = True
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)
    main(config)