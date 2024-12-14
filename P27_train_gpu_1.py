import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time



# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

# 数据集长度
train_set_size = len(train_data)
test_set_size = len(test_data)

print("训练数据集长度为:{}".format(train_set_size))     # 50000张
print("测试数据集长度为:{}".format(test_set_size))      # 10000张


# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


# 创建网络模型

tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

# 训练轮数
for i in range(epoch):
    print("---------第{}轮训练开始---------".format(i+1))

    # 训练步骤开始
    # tudui.train()     可以不加，只对特定层有作用(Dropout、BatchNorm)
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 首先优化器梯度清零
        optimizer.zero_grad()
        # 反向传播，计算损失函数对每个参数的梯度，供优化器更新时使用。
        loss.backward()
        # 据计算出的梯度，用优化算法(SGD等)更新模型参数，
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss:{}".format(total_train_step, loss.item()))     # loss.item()是标准写法,会把tensor数据类型转换成真实数字的写法
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    ''' 当上边的训练步骤结束以后，会进行下边的测试步骤测试网络 '''


    # 测试步骤开始
    # tudui.eval()  可以不加，只对特定层有作用(Dropout、BatchNorm)
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()  # loss.item()是标准写法,会把tensor数据类型转换成真实数字的写法
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i+1))
    # 方式2 torch.save(tudui.state_dirc(), "tudui_{}.path".faomar(i+1))
    print("模型已保存")

writer.close()






