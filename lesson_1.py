import torch
from pprint import pprint

one_dimensional_tensor = torch.tensor([2, 3])
two_dimensional_tensor = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32)
my_tensor = torch.tensor(
    [[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=torch.float32, requires_grad=True
)

print(f"Type of elements is {my_tensor.dtype}")  # Попертає тип елементів в тензорі
print(f"Tensor size = {my_tensor.shape}")  # Повертає розмір тензогу
print(f"Tensor size = {my_tensor.size()}")  # Повертає розмір тензогу
print(f"Quantity of axis = {my_tensor.ndim}")  # Кількість осей в тензорі
print(f"{my_tensor[0, 1, 1]=}")  # Доступ до елемента в тензорі
print(f"{my_tensor[0, 1, 1].item()=}")  # Отримання значення з улементу в тензорі

zeros_tensor = torch.zeros([3, 5])  # Строрює тензор розміром 3 на 5 заповнений нулями
print("Zeros tensor:")
pprint(zeros_tensor)

ones_tensor = torch.ones([3, 5])  # Строрює тензор розміром 3 на 5 заповнений oдиницями
print("Ones temsor:")
pprint(ones_tensor)

derivative_tensor = torch.zeros_like(
    ones_tensor
)  # Строрює тензор заповнений нулями розміром як батьківський (переданий) тензор
print("Derivative tensor:")
pprint(derivative_tensor)

full_like_tensor = torch.full_like(
    ones_tensor, 5
)  # Строрює тензор заповнений 5 розміром як батьківський (переданий) тензор
print("Full like tensor:")
pprint(full_like_tensor)

arange_tensor = torch.arange(
    2, 10, 0.5
)  # Створює тензор заповнений числами від 2 до 10 з кроком 0.5
pprint(arange_tensor)

first_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
print(first_tensor * 2)
my_list = [6, 7, 8, 9, 10]
second_tensor = torch.tensor(my_list, dtype=torch.float32)
print(first_tensor + second_tensor)

cpu_tensor = first_tensor.cpu()
print(f"CPU tensor: {cpu_tensor}")

# gpu_tensor = first_tensor.cuda()
# print(f"GPU tensor: {gpu_tensor}")

device = "cuda" if torch.cuda.is_available() else "cpu"
transformed_tensor = first_tensor.to(device)
print(f"Transformed tensor: {transformed_tensor}")
print(f"{transformed_tensor.device=}")
