import torch


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