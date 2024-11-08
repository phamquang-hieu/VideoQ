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
from sklearn.metrics import classification_report

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
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
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
    # model = model.cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        if config.TRAIN.SYMMETRIC_LOSS:
            criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        else:
            criterion = SoftTargetCrossEntropy()
            # criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                    smoothing=config.AUG.LABEL_SMOOTH,
                                    mixup_alpha=config.AUG.MIXUP, 
                                    cutmix_alpha=config.AUG.CUTMIX, 
                                    switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    # if config.TRAIN.OPT_LEVEL != 'O0':
    #     model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)


    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)


    text_labels = generate_text(train_data)
    if config.DATA.LABEL_2 is not None:
        import clip
        import pandas as pd
        classes_2 = pd.read_csv(config.DATA.LABEL_2).values.tolist()
        text_aug = f"{{}}"
        text_labels_2 = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in classes_2])
        text_id_2 = np.array([p[0] for p in classes_2])
    
    text_id = np.array([p[0] for p in train_data.classes])
    
    if config.TEST.ONLY_TEST:
        if config.DATA.LABEL_2 is not None:
            acc1 = validate_2stage(val_loader, text_labels, text_labels_2, text_id, text_id_2, model, config)
        else:
            acc1 = validate(val_loader, text_labels, text_id, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    # validate(train_loader, text_labels, text_id, model, config)
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, scaler)

        acc1 = validate(val_loader, text_labels, text_id, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        if dist.get_rank() == 0 and is_best:
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best=is_best)
        elif dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best=is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels, text_id, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn, scaler):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    y_true, y_pred = [], []
    acc1_meter = AverageMeter()
    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        with torch.cuda.amp.autocast(enabled=True):
            output, prompt_key_loss = model(images, texts)
            similarity = output.view(images.shape[0], -1).softmax(dim=-1)
            
            _, indices_1 = similarity.topk(1, dim=-1)

            ### eval during training ###
            acc1 = 0
            for i in range(images.shape[0]):
                y_pred.append(indices_1[i].cpu().item()), y_true.append(batch_data["label"][i].cpu().item())
                if indices_1[i].cpu().item() == batch_data["label"][i].cpu().item():
                    acc1 += 1

            acc1_meter.update(float(acc1) / images.shape[0] * 100, images.shape[0])
            ###############################

            if config.TRAIN.SYMMETRIC_LOSS:
                one_hot = torch.zeros((output.shape[1], output.shape[0])).to(output.device)
                for cnt in range(one_hot.shape[1]):
                    one_hot[batch_data["label"][cnt], cnt] = 1
                
                if config.AUG.LABEL_SMOOTH:
                    one_hot = one_hot*(1-config.AUG.LABEL_SMOOTH) + config.AUG.LABEL_SMOOTH/one_hot.shape[-1] 

                if isinstance(criterion, nn.KLDivLoss):
                    # label_id = nn.functional.one_hot(label_id, num_classes=config.DATA.NUM_CLASSES).to(torch.float32)
                    if config.AUG.LABEL_SMOOTH:
                        # label_id = label_id*(1-config.AUG.LABEL_SMOOTH) + config.AUG.LABEL_SMOOTH/label_id.shape[-1]
                        output = nn.functional.log_softmax(output, dim=-1)
                        label_id = label_id.log()
                        one_hot = one_hot.log()
                # print(label_id, label_id.shape)
                # print(criterion(output, label_id), criterion(output.t().contiguous(), one_hot))
                total_loss = 0.5*(criterion(output, label_id) + criterion(output.t().contiguous(), one_hot))
            else:
                if isinstance(criterion, nn.KLDivLoss):
                # label_id = nn.functional.one_hot(label_id, num_classes=config.DATA.NUM_CLASSES).to(torch.float32)
                    if config.AUG.LABEL_SMOOTH:
                        # label_id = label_id*(1-config.AUG.LABEL_SMOOTH) + config.AUG.LABEL_SMOOTH/label_id.shape[-1]
                        output = nn.functional.log_softmax(output, dim=-1)
                        label_id = label_id.log()
                total_loss = criterion(output, label_id)
            
            if prompt_key_loss is not None:
                total_loss += config.TRAIN.POOL_LAMBDA * prompt_key_loss

            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # if epoch < config.TRAIN.WARMUP_EPOCHS:
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            # if epoch < config.TRAIN.WARMUP_EPOCHS:
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'Acc@1: {acc1_meter.avg:.3f}')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f'\n{classification_report(y_true=y_true, y_pred=y_pred)}')


@torch.no_grad()
def validate(val_loader, text_labels, text_id:np.ndarray, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        y_true, y_pred = [], []
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES # number of frames in a video
            n = tn // t # number of views
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n): # for view in views
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                # if config.TRAIN.OPT_LEVEL == 'O2':
                #     image_input = image_input.half()
                with torch.cuda.amp.autocast(enabled=True):
                    output, _ = model(image_input, text_inputs)
                
                similarity = output.view(b, -1).softmax(dim=-1)
                similarity = sum_by_index(similarity, text_id, n_classes=config.DATA.NUM_CLASSES)
                tot_similarity += similarity # accumulating simmilarity from views
            # tot_similarity = sum_by_index(tot_similarity, indices=text_id, n_classes=14)s
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            if config.DATA.NUM_CLASSES > 5:
                values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                y_pred.append(indices_1[i].cpu().item()), y_true.append(label_id[i].cpu().item())

                if indices_1[i].cpu().item() == label_id[i].cpu().item():
                    acc1 += 1
                
                if config.DATA.NUM_CLASSES > 5:
                    if label_id[i].cpu().item() in indices_5[i].cpu():
                        acc5 += 1
           
            acc1_meter.update(float(acc1) / b * 100, b)
            if config.DATA.NUM_CLASSES > 5:
                acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f'\n{classification_report(y_true=y_true, y_pred=y_pred)}')
    return acc1_meter.avg

def sum_by_index(similarity: torch.Tensor, indices: np.ndarray, n_classes=14):
    result = torch.zeros([similarity.shape[0], n_classes], device=similarity.device)
    for b_id, b in enumerate(similarity):
        for i, item in enumerate(b):
            result[b_id, indices[i]] += item
    return result
    
@torch.no_grad()
def validate_2stage(val_loader, text_labels_1, text_labels_2, text_id_1:np.ndarray, text_id_2:np.ndarray, model, config):
    model.eval()
    # print(text_labels_2.shape)
    def views_inference(text_inputs, text_id, b, nd_stage):
        text_inputs = text_inputs

        if not nd_stage:
            image = _image[:, :, :, :, :, :] # [b,t,c,h,w]
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        else: 
            image = _image[b, :, :, :, :, :].unsqueeze(0)
            tot_similarity = torch.zeros((1, config.DATA.NUM_CLASSES)).cuda()
        for i in range(n): # for view in views
            # label_id = label_id.cuda(non_blocking=True)
            image_ = image[:, i, :, :, :, :]
            image_input = image_.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                output, _ = model(image_input, text_inputs)
            if not nd_stage:
                similarity = output.view(b, -1).softmax(dim=-1)
            else:
                similarity = output.view(1, -1).softmax(dim=-1)
            # print("tot_similarity shape", tot_similarity.shape)
            similarity = sum_by_index(similarity, text_id, config.DATA.NUM_CLASSES)
            tot_similarity += similarity # accumulating simmilarity from views
        return tot_similarity
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs_1 = text_labels_1.cuda()
        text_inputs_2 = text_labels_2.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        y_true, y_pred = [], []

        for idx, batch_data in enumerate(val_loader):
            # if idx < 104: continue
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES # number of frames in a video
            n = tn // t # number of views
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = views_inference(text_inputs=text_inputs_1, text_id=text_id_1, b=b, nd_stage=False)

            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            
            for i in range(b):
                gt_label = label_id[i].cpu().item()
                # if gt_label in text_id_1[indices_5[i].cpu()]:
                if gt_label in indices_5[i].cpu():
                    acc5 += 1

            acc5_meter.update(float(acc5) / b * 100, b)
            indices_5 = indices_5.cpu()
            # print("indices_5 before", indices_5)
            indices_5 = [np.unique(index) for index in indices_5]

            for i in range(b):
                mask = [index in indices_5[i] for index in text_id_2]
                text = text_inputs_2[mask]
                tot_similarity_2nd = views_inference(text_inputs=text, text_id=text_id_2[mask], b=i, nd_stage=True)
                values_1, indices_1 = tot_similarity_2nd.topk(1, dim=-1)
                gt_label = label_id[i].cpu().item()
                predicted = indices_1[0].cpu()

                y_true.append(gt_label), y_pred.append(int(predicted.numpy()))

                if gt_label == predicted:
                    acc1 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f'\n{classification_report(y_true=y_true, y_pred=y_pred)}')
    return acc1_meter.avg

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
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)