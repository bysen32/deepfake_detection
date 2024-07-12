from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from PIL import Image
import sys
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.datasets.funcs import load_json
import utils.misc as misc
import utils.lr_decay as lrd
from datetime import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import timm
from timm.loss import LabelSmoothingCrossEntropy
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR


from engine_finetune import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('SBI training and evaluation script', add_help=False)

    parser.add_argument('--image_size', default=380, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Model parameters
    parser.add_argument('--model', default='efficientnet_b4', type=str,
                        help='model name (default: efficientnet_b4)')
    

    # Optimizer parameters
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--blr', default=0.1, type=float,
                        help='base learning rate: absolute_lr = blr * eff_batch_size / 256')
    parser.add_argument('--layer_decay', default=0.75, type=float,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', default=5, type=int)

    # * fine-tuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_true', dest='global_pool',
                        help='use cls token instead of global pool for image classification')

    # Augmentation parameters
    parser.add_argument('--smoothing', default=0.1, type=float,)

    # Dataset parameters

    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)




    
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor)')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int,)
    parser.add_argument('--dist_on_itp', action='store_true',)
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    
    return parser



def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    misc.init_distributed_mode(args)

    print(f'job dir: {os.path.dirname(os.path.abspath(__file__))}')
    print("{}".format(args).replace(", ", "\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = SBI_Dataset(phase='train', image_size=args.image_size)
    # FIXME: 需要修改为正常验证数据集
    val_dataset   = SBI_Dataset(phase='val', image_size=args.image_size)


    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        print(f'Sampler_train = {sampler_train}')
        if args.dist_eval: # 推荐用于训练阶段快速评估
            if len(val_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter the evaluation results as extra duplicate samples are added to achieve '
                      'equal number of samples per process.')
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    else:
         sampler_train = torch.utils.data.RandomSampler(train_dataset)
         sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

   
    train_loader=torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=train_dataset.worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_loader=torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=val_dataset.worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=2)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print(f'number of params (M): {n_parameters/1.e6 :.2f}')
    
    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print(f'base lr: {args.lr * 256 / eff_batch_size:.2e}')
    print(f'actual lr: {args.lr:.2e}')

    print(f'effective batch size: {eff_batch_size}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print(f'critertion = {str(criterion)}')

    # resume
    # misc.load_model(args=args, model_without_ddp=model_without_ddp,
    #                 optimizer=optimizer, loss_scaler=loss_scaler)
    
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    max_accuracy = max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        max_auc = max(max_auc, test_stats['AUC'])

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            # log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # iter_loss=[]
    # train_losses=[]
    # test_losses=[]
    # train_accs=[]
    # test_accs=[]
    # val_accs=[]
    # val_losses=[]
    # n_epoch=args.epoch
    # lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    # last_loss=99999


    # now=datetime.now()
    # save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    # os.makedirs(save_path,exist_ok=True)
    # os.makedirs(save_path+'weights/',exist_ok=True)
    # os.makedirs(save_path+'logs/',exist_ok=True)
    # logger = log(path=save_path+"logs/", file="losses.logs")



    # # breakpoint()
    # last_auc=0
    # last_val_auc=0
    # weight_dict={}
    # n_weight=5
    # for epoch in range(n_epoch):
    #     np.random.seed(seed + epoch)
    #     train_loss=0.
    #     train_acc=0.
    #     model.train(mode=True)
    #     for step,data in enumerate(tqdm(train_loader)):
    #         img=data['img'].to(device, non_blocking=True).float()
    #         target=data['label'].to(device, non_blocking=True).long()
    #         output=model.training_step(img, target)
    #         loss=criterion(output,target)
    #         loss_value=loss.item()
    #         iter_loss.append(loss_value)
    #         train_loss+=loss_value
    #         acc=compute_accuray(F.log_softmax(output,dim=1),target)
    #         train_acc+=acc
    #     lr_scheduler.step()
    #     train_losses.append(train_loss/len(train_loader))
    #     train_accs.append(train_acc/len(train_loader))

    #     log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
    #                     epoch+1,
    #                     n_epoch,
    #                     train_loss/len(train_loader),
    #                     train_acc/len(train_loader),
    #                     )

    #     model.train(mode=False)
    #     val_loss=0.
    #     val_acc=0.
    #     output_dict=[]
    #     target_dict=[]
    #     np.random.seed(seed)
    #     for step,data in enumerate(tqdm(val_loader)):
    #         img=data['img'].to(device, non_blocking=True).float()
    #         target=data['label'].to(device, non_blocking=True).long()
    #         
    #         with torch.no_grad():
    #             output=model(img)
    #             loss=criterion(output,target)
    #         
    #         loss_value=loss.item()
    #         iter_loss.append(loss_value)
    #         val_loss+=loss_value
    #         acc=compute_accuray(F.log_softmax(output,dim=1),target)
    #         val_acc+=acc
    #         output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
    #         target_dict+=target.cpu().data.numpy().tolist()
    #     val_losses.append(val_loss/len(val_loader))
    #     val_accs.append(val_acc/len(val_loader))
    #     val_auc=roc_auc_score(target_dict,output_dict)
    #     log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
    #                     val_loss/len(val_loader),
    #                     val_acc/len(val_loader),
    #                     val_auc
    #                     )
    #  

    #     if len(weight_dict)<n_weight:
    #         save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
    #         weight_dict[save_model_path]=val_auc
    #         torch.save({
    #                 "model":model.state_dict(),
    #                 "optimizer":model.optimizer.state_dict(),
    #                 "epoch":epoch
    #             },save_model_path)
    #         last_val_auc=min([weight_dict[k] for k in weight_dict])

    #     elif val_auc>=last_val_auc:
    #         save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
    #         for k in weight_dict:
    #             if weight_dict[k]==last_val_auc:
    #                 del weight_dict[k]
    #                 os.remove(k)
    #                 weight_dict[save_model_path]=val_auc
    #                 break
    #         torch.save({
    #                 "model":model.state_dict(),
    #                 "optimizer":model.optimizer.state_dict(),
    #                 "epoch":epoch
    #             },save_model_path)
    #         misc.save_model(
    #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #             loss_scaler=loss_scaler, epoch=epoch)
    #         last_val_auc=min([weight_dict[k] for k in weight_dict])
    #     
    #     logger.info(log_text)
        
if __name__=='__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
