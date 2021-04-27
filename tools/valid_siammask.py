# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import torch
from torch.utils.data import DataLoader

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter

from datasets.siam_mask_dataset import DataSets

from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamMask in json format')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')


best_acc = 0.


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()

    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    
    print("Init logger")

    logger = logging.getLogger('global')

    print(44)
    #logger.info("\n" + collect_env_info())
    print(99)
    logger.info(args)

    cfg = load_config(args)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    print(2)

    if args.log_dir:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = Dummy()

    # build dataset
    train_loader, val_loader = build_data_loader(cfg)

    print(3)

    path = "/usr4/alg504/cliao25/siammask/experiments/siammask_base/snapshot/checkpoint_e{}.pth"

    for epoch in range(1,21):

        if args.arch == 'Custom':
            from custom import Custom
            model = Custom(pretrain=True, anchors=cfg['anchors'])
        else:
            exit()

        print(4)

        if args.pretrained:
            model = load_pretrain(model, args.pretrained)

        model = model.cuda()


        #model.features.unfix((epoch - 1) / 20)
        optimizer, lr_scheduler = build_opt_lr(model, cfg, args, epoch)
        filepath = path.format(epoch)
        assert os.path.isfile(filepath)

        model, _, _, _, _ = restore_from(model, optimizer, filepath)
        #model = load_pretrain(model, filepath)
        model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

        model.train()
        device = torch.device('cuda')
        model = model.to(device)

        valid(val_loader, model, cfg)

    print("Done")

def valid(val_loader, model, cfg):
    #model.train()
    #model = model.cuda()

    for iter, input in enumerate(val_loader):
        x = {
                  'cfg': cfg,
                  'template': torch.autograd.Variable(input[0]).cuda(),
                  'search': torch.autograd.Variable(input[1]).cuda(),
                  'label_cls': torch.autograd.Variable(input[2]).cuda(),
                  'label_loc': torch.autograd.Variable(input[3]).cuda(),
                  'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
                  'label_mask': torch.autograd.Variable(input[6]).cuda(),
                  'label_mask_weight': torch.autograd.Variable(input[7]).cuda(),
                 }
        outputs = model(x)

        # can't calculate rpn_loc_loss on validation set
        rpn_cls_loss, rpn_mask_loss = torch.mean(outputs['losses'][0]), torch.mean(outputs['losses'][1])
        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])

        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']

        print(rpn_cls_loss)
        print(mask_iou_mean)

        #loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight
        break
           


if __name__ == '__main__':
    main()
