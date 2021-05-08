import argparse

import numpy as np
import torch

from Model.SELayer import SEInception3


def main():
    parser = argparse.ArgumentParser(description="PyTorch MRI Training")
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--train', type=str, default='train',
                        help='put the path to resuming file if needed')
    args = parser.parse_args()
    model = SEInception3(2, aux_logits=False)
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    if args.train == 'train':
        md = np.load("dataloaders/md_train.npy")
        fa = np.load("dataloaders/fa_train.npy")
        mask = np.load("dataloaders/mask_train.npy")
    else:
        md = np.load("dataloaders/md_test.npy")
        fa = np.load("dataloaders/fa_test.npy")
        mask = np.load("dataloaders/mask_test.npy")
    md = torch.from_numpy(md).float()
    fa = torch.from_numpy(fa).float()
    mask = torch.from_numpy(mask).float()
    md = torch.unsqueeze(md, 1)
    fa = torch.unsqueeze(fa, 1)
    mask = torch.unsqueeze(mask, 1)
    image = torch.cat([md, fa, mask], 1)
    image = image.cuda()
    model.eval()
    with torch.no_grad():
        out = model(image)
    if args.train == 'train':
        torch.save(out, "predic.txt")
    else:
        torch.save(out, "predic2.txt")


if __name__ == "__main__":
    main()