"""
Class that handles data loading and preprocessing
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class BrainTumorDataset(Dataset):
    def __init__(self, path: str, img_size: int = 224, transform=None):
        self.path = path
        self.img_size = img_size
        self.transform = transform
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.data[idx][1]

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
def create_data_loader(path: str, batch_size: int = 64):
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

    image_datasets = {
        'train': BrainTumorDataset(path=path, transform=data_transforms['train']),
        'test': BrainTumorDataset(path=path, transform=data_transforms['test'])
    }

    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)
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