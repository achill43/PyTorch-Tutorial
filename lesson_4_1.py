import torch
import torchvision

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

# BETA-тестування torchvision-0,16
from torchvision.transforms import v2

#############################################################

import os

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

# Застосування трансформації до dataset
transform = v2.Compose(
    transforms=[
        v2.ToImage(),
        v2.Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)

train_data = ImageFolder(root="./mnist/training", transform=transform)
test_data = ImageFolder(root="./mnist/testing", transform=transform)

img_obj, cls_label = test_data[2]
print(f"{type(img_obj)=}")
print(f"{cls_label=}")
print(f"{type(img_obj)=}")
print(f"{img_obj.shape=}")  # (C, H, W)
print(f"{img_obj.dtype=}")
print(f"min = {img_obj.min()}; max = {img_obj.max()}")

# Split training data on training and validtion data

train_data, val_data = random_split(train_data, [0.8, 0.2])

# Create Batches
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
