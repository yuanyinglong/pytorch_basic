import torch
from torch import nn

# 搭建神经网络(因为数据集有10个类别，故网络应该是10分类网络)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )



    def forward(self, x):
        x = self.model(x)
        return x


# 验证神经网络是否正确
if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones(64, 3, 32, 32)
    output = tudui(input)
    print(output.shape)
