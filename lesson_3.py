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


train_data = ImageFolder(root="./mnist/training")
test_data = ImageFolder(root="./mnist/testing")


print(f"{train_data.classes=}")
print(f"{train_data.class_to_idx=}")
print(f"{len(train_data)=}")

img, one_hot_position = train_data[3810]  # get Image
cls = train_data.classes[one_hot_position]
print(f"Class: {cls}")
plt.imshow(img, cmap="gray")
plt.show()

for cls, one_hot_position in train_data.class_to_idx.items():
    one_hot_vector = [(i == one_hot_position) * 1 for i in range(10)]
    print(f"\033[32m{cls}\033[0m => \033[34m{one_hot_vector}\033[0m")


# Split training data on training and validtion data

train_data, val_data = random_split(train_data, [0.8, 0.2])

# Create Batches
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
