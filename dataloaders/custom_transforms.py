import torch
from torchsample.transforms import Rotate, RandomTranslate, Flip, ToTensor, Compose, RandomAffine
import numpy as np

class CustomRotate(object):
    def __init__(self, angle = 90):
        self.transforms = Compose([Rotate(angle)])
    def __call__(self, sample):
        
        md = np.array(sample['md'])
        fa = np.array(sample['fa'])
        mask = np.array(sample['mask'])
        label = sample['label']
        md = torch.from_numpy(md).float()
        for i in range(0, md.shape[0]) : md[i] = self.transforms(md[i])
        fa = torch.from_numpy(fa).float()
        for i in range(0, fa.shape[0]) : fa[i] = self.transforms(fa[i])
        mask = torch.from_numpy(mask).float()
        for i in range(0, mask.shape[0]) : mask[i] = self.transforms(mask[i])
        label = torch.Tensor([label]).long()
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        md = sample['md']
        fa = sample['fa']
        mask = sample['mask']
        label = sample['label']
        md = torch.from_numpy(md).float()
        fa = torch.from_numpy(fa).float()
        mask = torch.from_numpy(mask).float()
        label = torch.Tensor([label]).long()
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}