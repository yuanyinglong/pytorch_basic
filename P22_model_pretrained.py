import torchvision
from torch import nn

# 需要下载新数据集，看看的了!

# train_data = torchvision.datasets.ImageNet("./data_image_net", split="train", download='True',
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

vgg16_true.add_module('add_linear', nn.Linear(1000, 10))    # 添加一层
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[7] = nn.Linear(4096, 10)
print(vgg16_false)