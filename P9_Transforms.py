from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2
# 将图片进行修改并输出一个结果

# 用法
# 通过 transforms.Totensor解决两个问题


# 2、为什么需要Tensor数据类型

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")


# 1、transforms如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)
writer.close()


