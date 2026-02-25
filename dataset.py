__author__ = "Anil Appak"
__email__ = "ipekanilatalay@gmail.com"
__organization__ = "Tampere University"

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


class DeepDoFDataset(Dataset):
    def __init__(self, root_dir, raw_imsize):

        self.root_dir = root_dir
        self.raw_imsize = raw_imsize
   
        self.filenames = list(Path(root_dir).glob("*.png"))
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, ix):
        
        img = np.asarray(Image.open(self.filenames[ix]), dtype=np.float32) / 255.0

        # Convert the PIL image to a tensor
        img = transforms.ToTensor()(img)
        #img = img.permute(1, 2, 0)     
    
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.RandomVerticalFlip()(img)

        # Add random uniform noise
        img = img * (torch.rand(1) * 0.3 + 0.8)
        img = img / img.max()

        img_gt = img

        return img.float(), img_gt.float()



class DeepDoFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, raw_imsize, batch_size, num_workers):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.raw_imsize = raw_imsize
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Force CUDA initialization
        _ = torch.cuda.FloatTensor(1)
        self.train_data = DeepDoFDataset(self.data_dir / "train", raw_imsize=self.raw_imsize)
        self.valid_data = DeepDoFDataset(self.data_dir / "validation", raw_imsize=self.raw_imsize)
        self.test_data = DeepDoFDataset(self.data_dir / "test", raw_imsize=self.raw_imsize)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        pass


if __name__ == "__main__":
    dataset = DeepDoFDataset(root_dir=Path("./data/valid"))
    import IPython; IPython.embed(); exit(1)  # fmt: skip
