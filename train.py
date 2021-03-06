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
from Model.ResidualAttention.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from Model.ResNext.ResNeXt import ResNeXt



class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        self.train_loader = DataLoader(MRI_dataset(), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # Define channel num
        if args.data == 0:
            inchannel = 3
        elif 0 < args.data < 3:
            inchannel = 1
        else:
            inchannel = 2
         
        # Define network
        if args.model == 'res':
            model = ResidualAttentionModel()
        elif args.model == 'resnext':
            model = ResNeXt(cardinality=8,depth=29,nlabels=2,base_width=64)
        else:
            model = SEInception3(2, inchanel=inchannel, aux_logits=False)

        optimizer = torch.optim.SGD([{"params":model.parameters(),"initial_lr":args.lr}], lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # Define Criterion
        # whether to use class balanced weights
        if args.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.model, self.optimizer = model, optimizer

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda(self.args.gpu_ids[0])

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start_epoch is None:
                args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Define lr scheduler
        if args.start_epoch is None:
            args.start_epoch = 0
        if args.lr_scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_scheduler_gamma, last_epoch=args.start_epoch)
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma,
                                                                    last_epoch=args.start_epoch)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        train_correct = 0
        print('\n=>Epoches %i, learning rate = %.4f' % (epoch, self.scheduler.get_last_lr()[-1]))
        for i, sample in enumerate(tbar):
            md, fa, mask, target = sample['md'], sample['fa'], sample['mask'], sample['label']
            md = torch.unsqueeze(md, 1)
            fa = torch.unsqueeze(fa, 1)
            mask = torch.unsqueeze(mask, 1)
            if self.args.data == 0:
                image = torch.cat([md, fa, mask], 1)
            elif self.args.data == 1:
                image = md
            elif self.args.data == 2:
                image = fa
            elif self.args.data == 3:
                image = torch.cat([md, fa], 1)
            elif self.args.data == 4:
                image = torch.cat([md, mask], 1)
            elif self.args.data == 5:
                image = torch.cat([fa, mask], 1)
            target = torch.squeeze(target)
            if self.args.cuda:
                image, target = image.cuda(self.args.gpu_ids[0]), target.cuda(self.args.gpu_ids[0])
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            prediction = torch.max(output, 1)[1]
            train_correct = train_correct + (prediction == target).sum().item()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.scheduler.step()

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f, acc: %d' % (train_loss, train_correct))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MRI Training")
    parser.add_argument('--model', type=str, default='SENet',
                        choices=['SENet','res','resnext'],
                        help='model use')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'bce'],
                        help='loss func type (default: ce)')
    parser.add_argument('--data', type=int, default=0, metavar='N',
                        help='number of data type')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=None,
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
    parser.add_argument('--lr-scheduler', type=str, default='exp',
                        choices=['step', 'exp'],
                        help='lr scheduler mode: (default: exp)')
    parser.add_argument('--lr-scheduler-gamma', type=float, default=0.1,
                        help='the gamma of lr-scheduler')
    parser.add_argument('--lr-scheduler-step', type=str, default=None,
                        help='the mutistep')
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
    if args.lr_scheduler_step is not None:
        args.lr_scheduler_step = [int(s) for s in args.lr_scheduler_step.split(',')]
    if args.batch_size is None:
        args.batch_size = 8 * len(args.gpu_ids)
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