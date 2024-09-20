import os
import os.path as osp
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams['font.size'] = 15

class TimeEntry:
    def __init__(self, timestamp, function_name, time_consumed):
        self.timestamp = timestamp
        self.function_name = function_name
        self.time_consumed = time_consumed

# 读取函数数据
def read_function_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    realtime_limit = 1000 * float(lines[0].strip())
    header = lines[1].strip().split(',')
    header = [v.replace(" ", "") for v in header]

    entries = []
    for line in lines[10:]:
        parts = line.strip().split(',')
        for i, part in enumerate(parts[:-1]):
            if '=' in part:
                timestamp, value = part.split('=')
                try:
                    entry = TimeEntry(float(timestamp), header[i], float(value))
                    entries.append(entry)
                except ValueError:
                    continue
    return realtime_limit, header, entries

# 读取 GPU 数据
def read_gpu_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            timestamp, consumption = map(float, line.strip().split())
            data.append((timestamp, consumption))
    return data

# 筛选出重合的时间段
def find_overlapping_time(function_data, gpu_data):
    function_timestamps = [entry.timestamp for entry in function_data]
    gpu_timestamps = [timestamp for timestamp, _ in gpu_data]

    min_timestamp = max(min(function_timestamps), min(gpu_timestamps))
    max_timestamp = min(max(function_timestamps), max(gpu_timestamps))

    overlapping_function_data = [entry for entry in function_data if min_timestamp <= entry.timestamp <= max_timestamp]
    overlapping_gpu_data = [(timestamp, consumption) for timestamp, consumption in gpu_data if min_timestamp <= timestamp <= max_timestamp]

    return overlapping_function_data, overlapping_gpu_data

# 计算累积时间
def accumulate_time(entries, interval=0):
    accumulated_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    min_timestamp = float('inf')
    
    for entry in entries:
        if "verbose" not in entry.function_name:
            continue
        func_name = entry.function_name.replace("_verbose", "")
        min_timestamp = min(min_timestamp, entry.timestamp)
        
        rounded_timestamp = round(entry.timestamp / interval) * interval
        
        accumulated_data[func_name][rounded_timestamp][0] += entry.time_consumed
        accumulated_data[func_name][rounded_timestamp][1] += 1

    mean_data = defaultdict(lambda: defaultdict(float))
    for func, timestamps in accumulated_data.items():
        for ts, (total_time, count) in timestamps.items():
            if count > 0:
                mean_data[func][ts] = total_time / count
            else:
                mean_data[func][ts] = 0.0

    return mean_data, min_timestamp

# 计算 GPU 平均值
def compute_average_gpu_consumption(gpu_data, interval):
    averages = []
    current_bin_start = gpu_data[0][0]
    current_bin_end = current_bin_start + interval
    total_consumption = 0
    count = 0

    for timestamp, consumption in gpu_data:
        if timestamp < current_bin_end:
            total_consumption += consumption
            count += 1
        else:
            if count > 0:
                avg_consumption = total_consumption / count
                averages.append((current_bin_start, avg_consumption))
            else:
                averages.append((current_bin_start, None))

            current_bin_start = current_bin_end
            current_bin_end = current_bin_start + interval
            total_consumption = consumption
            count = 1

    # 处理最后一个区间
    if count > 0:
        avg_consumption = total_consumption / count
        averages.append((current_bin_start, avg_consumption))
    else:
        averages.append((current_bin_start, None))

    return averages

def plot_data(accumulated_data, min_timestamp, gpu_averages, interval=0.01, realtime_limit=100):
    x = sorted({timestamp for func_data in accumulated_data.values() for timestamp in func_data.keys()})
    x_adjusted = [(timestamp - min_timestamp) / interval for timestamp in x]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 设置颜色
    color_vec = generate_color_palette(len(accumulated_data))

    # 绘制实时限制线
    line_realtime_limit = ax1.axhline(y=realtime_limit, color='red', linestyle='--', linewidth=4, label='Real-time Limit')

    # 初始化底部叠加区域
    bottom = np.zeros(len(x_adjusted))

    # 绘制函数调用时间的叠加区域
    lines = []
    for i, (function_name, data) in enumerate(accumulated_data.items()):
        y = [data.get(timestamp, 0) for timestamp in x]
        line = ax1.fill_between(x_adjusted, bottom, bottom + y, color=color_vec[i], alpha=0.5, label=function_name)
        bottom += y
        lines.append(line)
    
    # 设置左侧y轴标签
    ax1.set_ylabel('Function Time (ms)')
    
    # 创建第二个y轴，并绘制GPU使用情况
    ax2 = ax1.twinx()
    gpu_x = [t for t, _ in gpu_averages]
    gpu_y = [c for _, c in gpu_averages if c is not None]
    
    # 调整GPU数据的长度与x_adjusted对齐
    min_length = min(len(gpu_x), len(x_adjusted))
    gpu_y, x_adjusted = gpu_y[:min_length], x_adjusted[:min_length]
    line_gpu = ax2.plot(x_adjusted, gpu_y, color='blue', linestyle='--', linewidth=1.5, label='GPU Usage')
    
    # 设置右侧y轴标签
    ax2.set_ylabel('GPU Usage (Mb)')
    ax2.set_ylim(bottom=0, top=max(gpu_y)*1)
    
    # 合并图例
    lines2 = [line_gpu[0], line_realtime_limit]
    lns = lines + lines2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')  # 设置图例位置为左上角

    # 设置x轴标签
    ax1.set_xlabel('Time (Interval: {}s)'.format(interval))
    
    # 设置网格和标题
    ax1.grid(linestyle=":", color="r")
# ax1.set_title('Time Consuming Filled Between Plot & GPU Memory Consuming on Botanic Garden Scequence 1018-13')
    plt.show()

def generate_color_palette(num_colors):
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_colors > len(colors):
        colors.extend(list(mcolors.BASE_COLORS.values()))
    if num_colors > len(colors):
        colors.extend(list(mcolors.CSS4_COLORS.values()))
    return colors[:num_colors]

def plot_box_plot(accumulated_data):
    filter_data = []
    label_box = []
    for function_name, data in accumulated_data.items():
        time_consumptions = list(data.values())
        filter_data.append(time_consumptions)
        label_box.append(function_name.replace("GS_", "").replace("_Update", "Expansion").replace("_", ""))
    
    plt.figure()
    boxplot = plt.boxplot(
        filter_data,
        medianprops={'color': 'red'},
        meanline=True,
        showmeans=True,
        meanprops={'color': 'blue', 'ls': '--'},
        flierprops={"marker": "*", "markerfacecolor": "red"},
        showfliers=True,
        patch_artist=True,
        boxprops={'facecolor': 'skyblue', 'edgecolor': 'black', 'linewidth': 2},
        labels=label_box
    )

    plt.grid(True)
    plt.legend(
        [boxplot["medians"][0], boxplot["means"][0], boxplot["fliers"][0]],
        ["Median", "Mean", "Outliers"],
        loc='upper right'
    )

    for i, function_name in enumerate(label_box[:1]):
        plt.annotate(
            "Median",
            xy=(i + 1, np.median(filter_data[i])),
            xytext=(15, 0),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            color='red'
        )

        plt.annotate(
            "Mean",
            xy=(i + 1, np.mean(filter_data[i])),
            xytext=(15, +15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            color='blue'
        )

        outliers = np.array(filter_data[i])
        if len(outliers) > 0:
            max_outlier = outliers[np.argmax(np.abs(outliers))]
            plt.annotate(
                "Outliers",
                xy=(i + 1, max_outlier),
                xytext=(15, -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"),
                color='red'
            )

# plt.title("Time Consuming Boxplot on Different Modules within One Iteration")
    plt.xlabel("Modules")
    plt.ylabel("Time (ms)")
    plt.show()

def main():
    function_filename = osp.join(sys.argv[1], "log_time.txt")
    gpu_filename = osp.join(sys.argv[1], "GPU.txt")
    interval = 1
    
    _, _, function_data = read_function_data(function_filename)
    gpu_data = read_gpu_data(gpu_filename)
    
    overlapping_function_data, overlapping_gpu_data = find_overlapping_time(function_data, gpu_data)
    
    accumulated_data, min_timestamp = accumulate_time(overlapping_function_data, interval=interval)
    gpu_averages = compute_average_gpu_consumption(overlapping_gpu_data, interval=interval)

    plot_data(accumulated_data, min_timestamp, gpu_averages, interval=interval)

    plot_box_plot(accumulated_data)

if __name__ == "__main__":
    main()
