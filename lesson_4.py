import torch
import torchvision

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

from torchvision import transforms

# BETA-тестування torchvision-0,16
from torchvision.transforms import v2

#############################################################

import os

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image


plt.axis("off")
plt.imshow(Image.open("./img/test.jpeg"))
plt.show()

img = np.array(Image.open("./img/test.jpeg"))
print(f"{type(img)=}")
print(f"{img.shape=}")  # (H, W, C)
print(f"{img.dtype=}")
print(f"min = {img.min()}; max = {img.max()}")


# Перетворення зображення в torch.tensor()
# (PIL.image, np.array) ==> torch.tensor
# (H, W, C) ==> (C, H, W)
# (0, 255) ==> (0.0, 1.0)
transform = transforms.ToTensor()
img_to_tensor = transform(img)
print(f"{type(img_to_tensor)=}")
print(f"{img_to_tensor.shape=}")  # (C, H, W)
print(f"{img_to_tensor.dtype=}")
print(f"min = {img_to_tensor.min()}; max = {img_to_tensor.max()}")

# Нормалізація даних
transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
img_normaliz = transform(img_to_tensor)
print(f"{type(img_normaliz)=}")
print(f"{img_normaliz.shape=}")  # (C, H, W)
print(f"{img_normaliz.dtype=}")
print(f"min = {img_normaliz.min()}; max = {img_normaliz.max()}")


# Послідовне застосування трансформацій
transform = transforms.Compose(
    transforms=[
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
img_tensor = transform(Image.open("./img/test.jpeg"))
print(f"{type(img_tensor)=}")
print(f"{img_tensor.shape=}")  # (C, H, W)
print(f"{img_tensor.dtype=}")
print(f"min = {img_tensor.min()}; max = {img_tensor.max()}")


# Перетворення зображення в torch.tensor()
# (PIL.image, np.array) ==> torch.tensor
# (H, W, C) ==> (C, H, W)
# (0, 255) ==> (0.0, 1.0)
transform = v2.ToTensor()
img_v2_tensor = transform(Image.open("./img/test.jpeg"))
print(f"{type(img_v2_tensor)=}")
print(f"{img_v2_tensor.shape=}")  # (C, H, W)
print(f"{img_v2_tensor.dtype=}")
print(f"min = {img_v2_tensor.min()}; max = {img_v2_tensor.max()}")


# Перетворення зображення в torchvision.tv_tensors.
# (PIL.image, np.array) ==> torchvision.tv_tensors
# (H, W, C) ==> (C, H, W)
# (0, 255) ==> (0.0, 1.0)

transform = v2.ToImage()
img_dtype_tensor = transform(Image.open("./img/test.jpeg"))
print(f"{type(img_dtype_tensor)=}")
print(f"{img_dtype_tensor.shape=}")  # (C, H, W)
print(f"{img_v2_tensor.dtype=}")
print(f"min = {img_dtype_tensor.min()}; max = {img_dtype_tensor.max()}")


# Послідовне застосування трансформацій
transform = v2.Compose(
    transforms=[
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
img_final_tensor = transform(Image.open("./img/test.jpeg"))
print(f"{type(img_tensor)=}")
print(f"{img_final_tensor.shape=}")  # (C, H, W)
print(f"{img_final_tensor.dtype=}")
print(f"min = {img_final_tensor.min()}; max = {img_final_tensor.max()}")

"""
    Послідовність застосування трансформацій:
    transform = v2.Compose(
        transforms=[
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            --- Додавання потрібних переьворень: 
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
)
"""

# Застосування трансформації до dataset
transform = v2.Compose(
    transforms=[
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

train_data = ImageFolder(root="./mnist/training", transform=transform)
test_data = ImageFolder(root="./mnist/testing", transform=transform)

img_obj, cls_label = test_data[2]
print(f"{img_obj=}")
print(f"{cls_label=}")
print(f"{type(img_obj)=}")
print(f"{img_obj.shape=}")  # (C, H, W)
print(f"{img_obj.dtype=}")
print(f"min = {img_obj.min()}; max = {img_obj.max()}")
