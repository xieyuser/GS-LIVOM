import os
import os.path as osp
import cv2
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

def concat_images(img1, img2):
    """将两张图片横向拼接"""
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst

def create_video_from_images(folder1, folder2, output_video_path, fps=10):
    """
    从两个文件夹中的图片创建视频
    :param folder1: 第一个文件夹路径
    :param folder2: 第二个文件夹路径
    :param output_video_path: 输出视频路径
    :param fps: 视频帧率，默认为30
    """
    # 获取所有图片文件名
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.png')], key=lambda x: int(osp.basename(x).split('.')[0]))
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.png')], key=lambda x: int(osp.basename(x).split('.')[0]))

    # 确保两个文件夹中的图片数量相同
    llen = min(len(files1), len(files2))

    # 读取第一张图片以获取尺寸
    img1 = Image.open(os.path.join(folder1, files1[0]))
    img2 = Image.open(os.path.join(folder2, files2[0]))
    video_size = (img1.width + img2.width, max(img1.height, img2.height))

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    # 遍历所有图片并拼接后写入视频
    for file1, file2 in tqdm(zip(files1[6:llen], files2[:llen-6]), total=llen):
        img1 = Image.open(os.path.join(folder1, file1))
        img2 = Image.open(os.path.join(folder2, file2))
        combined_img = concat_images(img1, img2)
        # 将PIL图像转换为OpenCV格式
        frame = cv2.cvtColor(np.array(combined_img), cv2.COLOR_RGB2BGR)
        out.write(frame)

    # 释放资源
    out.release()

# 使用示例
folder1 = sys.argv[1]
folder2 = sys.argv[2]
output_video_path = 'output.mp4'
create_video_from_images(folder1, folder2, output_video_path)