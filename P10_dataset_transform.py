import torchvision
from torch.utils.tensorboard import SummaryWriter

# Compose()可按顺序将多种变换组合到一起。
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset1",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset1",train=False,transform=dataset_transform,download=True)

# print(test_set[0])
# print(test_set.classes)     # 输出类
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0])

writer = SummaryWriter(log_dir='p10')
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()



