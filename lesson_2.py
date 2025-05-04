import torch
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

##############################################################

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.tranform = transform

        self.len_dataset = 0
        self.data_list = list()

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split("/")[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset = len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = np.array(Image.open(file_path))

        if self.tranform is not None:
            sample = self.tranform(sample)

        return sample, target


train_dataset = MNISTDataset(path="./mnist/training")
test_dataset = MNISTDataset(path="./mnist/testing")

print(f"{train_dataset.classes=}")
print(f"{train_dataset.class_to_idx=}")

for cls, one_hot_position in train_dataset.class_to_idx.items():
    one_hot_vector = [(i == one_hot_position) * 1 for i in range(10)]
    print(f"\033[32m{cls}\033[0m => \033[34m{one_hot_vector}\033[0m")

print(f"{len(train_dataset)=}")

img, one_hot_position = train_dataset[3810]  # get Image
cls = train_dataset.classes[one_hot_position]
print(f"Class: {cls}")
plt.imshow(img, cmap="gray")
plt.show()

# Split training data on training and validtion data

train_data, val_data = random_split(train_dataset, [0.8, 0.2])

# Create Batches
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
