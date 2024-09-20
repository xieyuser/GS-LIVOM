#!/usr/bin/env python
import os
import os.path as osp
import shutil

cur_d = osp.dirname(__file__)

import rospy
from nav_msgs.msg import Odometry

import subprocess
import time

def get_gpu_memory_usage():
    try:
        # Run the nvidia-smi command to query GPU memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        if result.returncode != 0:
            raise Exception(f"nvidia-smi command failed: {result.stderr}")
        
        # Parse the output and convert to float (in MiB)
        memory_usages = [float(x) for x in result.stdout.strip().split('\n')]
        
        return memory_usages[0]

    except Exception as e:
        print(f"Error while fetching GPU memory usage: {e}")
        return False

def callback(data):
    # 提取时间戳
    timestamp = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9
    
    # 提取位置和姿态
    position = data.pose.pose.position
    orientation = data.pose.pose.orientation
    
    # 构造输出字符串
    output_str = "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
        timestamp, position.x, position.y, position.z,
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    
    with open(save_odom_path, "a") as file:
        file.write(output_str)
    
    with open(save_gpu_path, "a") as file:
        mem = get_gpu_memory_usage()
        if mem:
            file.write("{:.6f} {:.2f}\n".format(
                rospy.Time.now().to_sec(),
                mem
            ))

def mktree(pkg_path, save_path):
    # 构建完整路径
    root_dir = osp.join(pkg_path, f"./output/{save_path}")
    print("root_dir: ", root_dir)

    if osp.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    shutil.copytree(osp.join(pkg_path, f"./output/sparse"), osp.join(root_dir, f"./sparse"))

    rendered_images_dir = osp.join(root_dir, "rendered_images")
    rendered_depths_dir = osp.join(root_dir, "rendered_depths")

    os.makedirs(rendered_images_dir, exist_ok=True)
    os.makedirs(rendered_depths_dir, exist_ok=True)

if __name__ == '__main__':
    rospy.init_node('odom_listener', anonymous=True)
    bag_path = rospy.get_param('bag_path')
    rospy.loginfo("Bag path: %s", bag_path)
    save_path = bag_path.split("/")[-1][:-4]
    pkg_path = osp.join(cur_d, "../../../")

    mktree(pkg_path, save_path)

    save_odom_path = osp.join(pkg_path, f"./output/{save_path}/Ours.txt")
    print("odom saved in: " , save_odom_path)
    
    save_gpu_path = osp.join(pkg_path, f"./output/{save_path}/GPU.txt")
    print("odom saved in: " , save_gpu_path)

    with open(save_odom_path, 'w') as file:
        file.write('')

    with open(save_gpu_path, 'w') as file:
        file.write('')

    # lvi sam
    # rospy.Subscriber("/odometry/imu", Odometry, callback)
    # fast lio and gslivom
    rospy.Subscriber("/Odometry", Odometry, callback)
    # r3live
    # rospy.Subscriber("/camera_odom", Odometry, callback)    
    rospy.spin()
