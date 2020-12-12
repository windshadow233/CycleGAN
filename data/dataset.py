import glob
import os
from PIL import Image
import random
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.img_A = glob.glob(os.path.join(root, f'{mode}A', '*.jpg'))
        self.img_B = glob.glob(os.path.join(root, f'{mode}B', '*.jpg'))
        self.transform = transform

    def __getitem__(self, item):
        img_A = Image.open(self.img_A[item % len(self.img_A)])
        img_B = Image.open(self.img_B[random.randint(0, len(self.img_B) - 1)])
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B

    def __len__(self):
        return max(len(self.img_A), len(self.img_B))
