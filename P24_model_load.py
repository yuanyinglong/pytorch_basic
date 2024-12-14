import torch
import torchvision
from torch import nn

# 方式1->保存方式1，加载模型
model1 = torch.load("vgg16_method1.pth")
print("模型1：", model1)


# 方式2->保存方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)  # 新建网络模型结构
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # 获取参数
# model = torch.load("vgg16_method2.pth")
print("模型2：", vgg16)

# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model3 = torch.load("tudui_method1.pth")
print("模型3：", model3)





