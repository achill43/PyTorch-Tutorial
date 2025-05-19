# Функція втрат
import torch
import torch.nn as nn


# Create Neuron Network Model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

input = torch.rand([16, 784], dtype=torch.float32)
output = model(input)
print(f"{output.shape=}")


# model_state = model.state_dict()    # Return model state dict
# print(f"{model_state.get('0.weight')=}")
# print(f"{model_state.get('0.bias')=}")


# for parameter in model.parameters():
#     print(f"{parameter=}")
#     print(f"{parameter.shape=}", end="\n\n")


# model.train()
# model.eval()


model_2 = nn.Sequential()
model_2.add_module("layer_1", nn.Linear(784, 128))
model_2.add_module("relu", nn.ReLU())
model_2.add_module("layer_2", nn.Linear(128, 10))
print(f"{model_2.layer_1=}")
input = torch.rand([16, 784], dtype=torch.float32)
output_2 = model_2(input)
print(f"{output_2.shape=}")
