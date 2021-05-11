import torch
import numpy as np
import random

class RamdomRotate(object):

    def __call__(self, sample):
        
        md = sample['md']
        fa = sample['fa']
        mask = sample['mask']
        label = sample['label']
        angle = random.randint(0, 3)
        axe = random.randint(0, 2)
        if axe == 0:
            axes = (0, 1)
        elif axe == 1:
            axes = (1, 2)
        elif axe == 2:
            axes = (0, 2)
        md = np.rot90(md, angle, axes=axes)
        fa = np.rot90(fa, angle, axes=axes)
        mask = np.rot90(mask, angle, axes=axes)
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}

class RamdomFlip(object):

    def __call__(self, sample):
        
        md = sample['md']
        fa = sample['fa']
        mask = sample['mask']
        label = sample['label']
        angle = random.randint(0, 2)
        md = np.flip(md, angle)
        fa = np.flip(fa, angle)
        mask = np.flip(mask, angle)
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}
    
class RamdomMask(object):

    def __call__(self, sample):
        
        md = np.array(sample['md'])
        fa = np.array(sample['fa'])
        mask = np.array(sample['mask'])
        label = sample['label']
        prob1 = random.randint(0, 1)
        prob2 = random.randint(0, 1)
        if prob1 == 0:
            md = md * mask
        if prob2 == 0:
            fa = fa * mask
        if prob1 == 1 and prob2 == 1:
            ones = np.where(mask == 0)
            positions = []
            for i in ones :
                positions.append((-min(i), mask.shape[0] - 1 - max(i)))
            for i in range(0, len(positions)) : 
                shift = random.randint(positions[i][0], positions[i][1])
                md = np.roll(md, shift, i)
                fa = np.roll(fa, shift, i)
                mask = np.roll(mask, shift, i)
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        md = np.ascontiguousarray(sample['md'])
        fa = np.ascontiguousarray(sample['fa'])
        mask = np.ascontiguousarray(sample['mask'])
        label = sample['label']
        md = torch.from_numpy(md).float()
        fa = torch.from_numpy(fa).float()
        mask = torch.from_numpy(mask).float()
        label = torch.Tensor([label]).long()
        return {'md': md, 'fa': fa, 'mask': mask, "label": label}
        
