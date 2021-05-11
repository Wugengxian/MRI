from torch.utils.data import Dataset
import numpy as np
from numpy import random
from torchvision.transforms import transforms
import dataloaders.custom_transforms as tr
from torch.utils.data import DataLoader


class MRI_dataset(Dataset):
    def __init__(self):
        super().__init__()

        self.md = np.load("dataloaders/md_train.npy")
        self.fa = np.load("dataloaders/fa_train.npy")
        self.mask = np.load("dataloaders/mask_train.npy")
        self.label = np.load("dataloaders/label_train.npy")
        self.real_length = self.label.shape[0]
        self.zeros = np.where(self.label == 0)
        self.oversampled_length = self.real_length + 3 * len(self.zeros[0])

    def __len__(self):
        return self.oversampled_length

    def __getitem__(self, index):
        real_index = index % self.oversampled_length
        if(real_index >= self.real_length) : real_index = self.zeros[0][(real_index - self.real_length) % len(self.zeros[0])]
        md, fa, mask, label = self.md[real_index], self.fa[real_index], self.mask[real_index], self.label[real_index]
        sample = {'md': md, 'fa': fa, "mask": mask, "label": label}

        return self.transform_tr(sample)

    def transform_tr(self, sample, tr_type:int = 0):

        composed_transforms = transforms.Compose([tr.RamdomMask(), tr.RamdomFlip(), tr.RamdomRotate(), tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == "__main__":
    data = DataLoader(MRI_dataset(), batch_size=1, shuffle=True, drop_last=False)
    pass