import os

import numpy as np
import torch
from PIL import Image
import argparse
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader

from Model.SELayer import se_inception_v3, SEInception3
from Saver.saver import Saver
from dataloaders.MRI_dataset import MRI_dataset
from Model.ResNext.ResNeXt import ResNeXt

# device_ids = ["7"]
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.device_comp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        self.train_loader = DataLoader(MRI_dataset(), batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Define network
        self.model = ResNeXt(cardinality=8,depth=29,nlabels=2,base_width=64)
        if args.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        optimizer = torch.optim.SGD([{"params":self.model.parameters(),"initial_lr":args.lr}], lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # Define Criterion
        # whether to use class balanced weights
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

        # Using cuda

            # self.model = self.model.to(device=self.device_comp)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Define lr scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8, last_epoch=args.start_epoch)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        print('\n=>Epoches %i, learning rate = %.4f' % (epoch, self.scheduler.get_last_lr()[-1]))
        for i, sample in enumerate(tbar):
            md, fa, mask, target = sample['md'], sample['fa'], sample['mask'], sample['label']
            md = torch.unsqueeze(md, 1)
            fa = torch.unsqueeze(fa, 1)
            mask = torch.unsqueeze(mask, 1)
            image = torch.cat([md, fa, mask], 1)
            target = torch.squeeze(target)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.scheduler.step()

            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MRI Training")
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['vgg16'],
                        help='model use')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                                comma-separated list of integers only (default=0)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='MRI',
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.batch_size is None:
        args.batch_size = 4
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.lr is None:
        args.lr = 0.05 / (8 * len(args.gpu_ids)) * args.batch_size
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        #     trainer.validation(epoch, args)
    # trainer.writer.close()


if __name__ == "__main__":
    main()
