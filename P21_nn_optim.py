import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):

        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()    # 交叉熵损失
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)    # 定义一个随机梯度下降（SGD）优化器，使用指定的学习率 lr=0.01

for epoch in range(20):
    running_loss = 0.0      # 初始化当前 epoch 中，累计损失的变量。
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)    # 当前样本的损失
        optim.zero_grad()       # 清空优化器中模型参数的梯度，避免前一批次的梯度对当前梯度的累积。
        result_loss.backward()  # 反向传播，计算损失函数对每个参数的梯度。(梯度会存储在对应参数的 grad 属性中，供优化器更新时使用。)
        optim.step()            # 根据计算出的梯度更新模型参数。(利用优化器的算法（如 SGD、Adam 等）进行参数更新。)
        running_loss = running_loss + result_loss   # 整体loss总和
    print(running_loss)








