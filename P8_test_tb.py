from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')  # 创建一个日志文件夹，名为logs
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)    # 将图片加载成为一个PIL图像对象
img_array = np.array(img_PIL)   # 将 PIL 图像对象转换为一个 NumPy 数组，通常形状为 (H, W, C)。(高，宽，通道数)
print(img_array.shape)


writer.add_image("test",img_array,2,dataformats='HWC')


# 绘图y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x",2*i,i)

writer.close()






