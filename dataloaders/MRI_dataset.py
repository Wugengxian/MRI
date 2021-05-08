from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
import dataloaders.custom_transforms as tr


class MRI_dataset(Dataset):
    def __init__(self):
        super().__init__()

        self.md = np.load("dataloaders/md_train.npy")
        self.fa = np.load("dataloaders/fa_train.npy")
        self.mask = np.load("dataloaders/mask_train.npy")
        self.label = np.load("dataloaders/label_train.npy")

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        md, fa, mask, label = self.md[index], self.fa[index], self.mask[index], self.label[index]
        sample = {'md': md, 'fa': fa, "mask": mask, "label": label}

        return self.transform_tr(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()])

        return composed_transforms(sample)