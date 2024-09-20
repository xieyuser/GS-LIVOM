import os
import rosbag
from geometry_msgs.msg import PoseStamped
import sys

def extract_pose_from_bag(bag_file_path, output_file_path):
    """
    从给定的bag文件中提取PoseStamped消息，并将它们转换为TUM格式。
    """
    # 创建输出文件
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # 打开bag文件
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
            # 提取时间戳
            timestamp = t.to_sec()
            
            # 提取位置和姿态
            position = msg.pose.position
            orientation = msg.pose.orientation
            
            # 将数据写入文件
            with open(output_file_path, 'a') as file:
                file.write(f"{timestamp} {position.x} {position.y} {position.z} "
                           f"{orientation.x} {orientation.y} {orientation.z} {orientation.w}\n")

def process_bags_in_folder(folder_path):
    """
    遍历文件夹中的所有bag文件，并对每个文件调用extract_pose_from_bag函数。
    """
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.bag'):
            # 构建完整的文件路径
            bag_file_path = os.path.join(folder_path, filename)
            
            # 构建输出文件名
            output_file_name = os.path.splitext(filename)[0] + '.txt'
            output_file_path = os.path.join(folder_path, output_file_name)
            
            # 调用函数处理当前bag文件
            extract_pose_from_bag(bag_file_path, output_file_path)

if __name__ == '__main__':
    # 指定包含bag文件的文件夹路径
    folder_path = sys.argv[1]
    
    # 调用函数处理文件夹中的所有bag文件
    process_bags_in_folder(folder_path)