import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        '''# 此处被Sequential代替
        self.conv1 = Conv2d(3, 32, 5, padding=2)    # stride和padding是计算出来的
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()        # 展平
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
        '''

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
        '''被下方x = self.model1(x)所代替
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        '''
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()    # 交叉熵损失
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()      # 是上一步求出来的变量用backward(),而不是loss用backward()
    print("ok {}".format(result_loss.item()))








