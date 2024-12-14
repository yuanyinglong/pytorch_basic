import torch
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs,(1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)

loss_mes = MSELoss()
result_mse = loss_mes(inputs, targets)
print(result)
print(result_mse)




