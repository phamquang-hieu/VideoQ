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
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
# from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
import gc

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

def main(config):
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=None,
                        )
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)
    model.eval()
    model.cuda()

    if config.INFER.MODEL: #PQH: if the checkpoint used for inference is specified
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer=None, lr_scheduler=None, logger=logger)

    # 3 first step: split the video into segments of 10 seconds
    #         dict(type='DecordInit'),
    #         dict(type='SampleAnnotatedFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
    #         dict(type='DecordDecode'),
    # 
    # val_pipeline = [
            
    #         dict(type='Resize', scale=(-1, scale_resize)),
    #         dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='FormatShape', input_format='NCHW'),
    #         dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    #         dict(type='ToTensor', keys=['imgs'])
    #     ]
    


if __name__ == "__main__":
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

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    main(config=config)
