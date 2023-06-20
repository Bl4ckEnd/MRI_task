"""
Class that handles data loading and preprocessing
"""
import copy
from typing import Optional
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random


class BrainTumorDataset(Dataset):
    def __init__(self, path: str, img_size: int = 224, transform=None, start_idx: int = 0, length: Optional[int]=None):
        self.path = path
        self.img_size = img_size
        self.transform = transform
        self.data = self._load_data()
        # shuffle data for splitting
        random.shuffle(self.data)
        self.length = length or len(self.data)
        self.start_idx = start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.start_idx
        img_path, label = self.data[idx]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def _load_data(self):
        data = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".jpg"):
                    label = 0 if "no" in file else 1
                    data.append((os.path.join(root, file), label))
        return data


# create data loader
def create_data_loader(path: str, batch_size: int = 10, split=(0.6, 0.2, 0.2)):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
    }
    # we need the same dataset as we randomly shuffle, TODO: nicer if we use subset
    train_dataset = BrainTumorDataset(path=path)
    val_dataset = copy.deepcopy(train_dataset)
    test_dataset = copy.deepcopy(train_dataset)
    data_len = len(train_dataset)

    train_dataset.length = int(data_len * split[0])
    train_dataset.transform = data_transforms["train"]

    val_dataset.length = int(data_len * split[1])
    val_dataset.start_idx = train_dataset.length
    val_dataset.transform = data_transforms["test"]

    test_dataset.length = data_len - val_dataset.length - train_dataset.length
    test_dataset.start_idx = val_dataset.length + train_dataset.length
    test_dataset.transform = data_transforms["test"]
    

    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size),
        'test': DataLoader(test_dataset, batch_size=batch_size)
    }
    return data_loaders


if __name__ == '__main__':
    dataset = BrainTumorDataset(path='data/brain_tumor_dataset')
    print(dataset[0])
    print(len(dataset))

    data_loader = create_data_loader(path='data/brain_tumor_dataset')
    print(len(data_loader['train']))
    print(len(data_loader['test']))
    print(data_loader['train'].dataset[0])