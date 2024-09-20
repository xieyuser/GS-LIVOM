import rosbag
import cv2
import numpy as np
import os
import sys
from cv_bridge import CvBridge

def extract_compressed_image(bag_file_path, output_image_dir, output_text_path):
    """
    从给定的bag文件中提取CompressedImage消息，并将它们保存为PNG文件。
    同时记录时间戳和文件名到文本文件中。
    """
    # 创建输出文件夹
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    
    # 清空输出文本文件
    if os.path.exists(output_text_path):
        os.remove(output_text_path)


 # 创建CvBridge实例
    bridge = CvBridge()

    # 打开bag文件
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/dalsa_rgb/left/image_raw']):
            # 获取时间戳
            timestamp = t.to_sec()
            
            # 解压缩图像
            # np_arr = np.frombuffer(msg.data, np.uint8)
            # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

             # 将Image消息转换为OpenCV图像
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 生成文件名
            filename = f"{timestamp}.png"
            
            # 保存图像
            cv2.imwrite(os.path.join(output_image_dir, filename), img)
            
            # 将数据写入文本文件
            with open(output_text_path, 'a') as file:
                file.write(f"{timestamp} rgb/{filename}\n")

if __name__ == '__main__':
    # 指定bag文件路径
    bag_file_path = sys.argv[1]
    # 指定输出文件夹路径
    output_image_dir = 'rgb/'
    # 指定输出文本文件路径
    output_text_path = 'rgb.txt'
    
    # 调用函数
    extract_compressed_image(bag_file_path, output_image_dir, output_text_path)