import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from moviepy.editor import ImageSequenceClip
import logging

# 设置字体和日志级别
plt.rcParams["font.family"] = 'SimHei'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_age_group_over_years(df, age_group):
    """
    绘制指定年龄段随年份变化的人口统计纵向条形图，并返回图像数组。
    """
    years = df.columns.tolist()
    population_counts = df.loc[age_group].tolist()

    fig, ax = plt.subplots(figsize=(9, 16))

    # 创建一个从绿色到红色的渐变色彩映射
    cmap = plt.get_cmap('plasma')  # 使用预定义的色彩映射，如'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    
    # 创建一个线性的渐变颜色列表，长度等于条形的数量
    gradient = np.linspace(0, 1, len(population_counts))
    colors = cmap(gradient)

    # 绘制条形图并应用渐变色
    bars = ax.bar(years, population_counts, color=colors)


    # ax.bar(years, population_counts, color='green')
    ax.set_ylabel('人口数', fontsize=18, color='gray')
    ax.set_xlabel('年份', fontsize=18, color='gray')
    ax.set_title(f'{age_group} 年龄段人口统计', fontsize=30, color='white', fontweight='bold',bbox=dict(facecolor='red', edgecolor='none', pad=8.0),pad=20)

    # 调整纵坐标范围并标注数值
    ax.set_ylim(0, max(population_counts) * 1.1)
    for i, v in enumerate(population_counts):
        ax.text(i, v + 3, f"{v}", color='blue', fontsize=10, fontweight='bold', ha='center')

    # 保存图像为数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def generate_video_from_single_csv(file_path):
    all_frames = []

    # 读取整个CSV文件并处理数据
    df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

    # 获取所有的年龄段
    age_groups = df.index.tolist()

    for age_group in age_groups:
        frame = plot_age_group_over_years(df, age_group)
        # 将当前帧重复添加到帧列表中
        all_frames.extend([frame] * 6)

    # 创建视频但不包括最后两年的数据
    clip = ImageSequenceClip(all_frames, fps=1)
    clip.write_videofile('population_video.mp4', codec='libx264')

if __name__ == "__main__":
    csv_file_path = 'Population_age_distribution_Changes_Over_Years.csv'
    generate_video_from_single_csv(csv_file_path)


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from moviepy.editor import ImageSequenceClip
# import os
# import logging

# # 设置字体
# plt.rcParams["font.family"] = 'SimHei'

# # 设置日志级别
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.INFO)

# # 设置日志记录器
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# def read_csv_and_process(file_path):
#     """
#     读取CSV文件并返回处理后的人口统计数据（年龄段和人口数以及年份）
#     """
#     df = pd.read_csv(file_path, index_col=0, encoding='utf-8')
#     age_groups = df.index.tolist()
#     population_counts = df.iloc[:, 0].tolist()  # 假设每列代表一年的数据，这里只选取第一列
#     year = os.path.splitext(os.path.basename(file_path))[0]
#     logging.debug(f"Processing data for year: {year}")
#     return list(zip(age_groups, population_counts, [year]*len(age_groups)))


# def plot_vertical_bar(age_groups, population_counts, year):
#     """
#     绘制纵向条形图展示人口统计数据，并标注年份，返回图像数组
#     """
#     fig, ax = plt.subplots(figsize=(20, 12))  # 调整图表尺寸以适应竖直的条形图
#     ax.bar(age_groups, population_counts)     # 使用 bar() 方法创建垂直条形图
#     ax.set_ylabel('人口数')                     # 交换x轴和y轴标签
#     ax.set_xlabel('年龄段')
#     ax.set_title(f'{year}年人口统计',fontsize=16, color='white', fontweight='bold',bbox=dict(facecolor='red', edgecolor='none', pad=8.0),pad=20)

#     # 调整坐标轴范围，为顶端预留空间用于显示年份
#     ax.set_ylim(0, max(population_counts) * 1.4)

#     for i, v in enumerate(population_counts):
#         ax.text(i, v + 0.05, f"{v}", color='black', fontweight='bold', ha='center')
#         # ax.text(i, max(population_counts) * 1.2, year, color='gray', fontsize=15, ha='center', va='bottom')

#     # 保存图像为数组
#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return image


# def plot_horizontal_bar(age_groups, population_counts, year):
#     """
#     绘制横向条形图展示人口统计数据，并标注年份，返回图像数组
#     """
#     fig, ax = plt.subplots(figsize=(10, 12))
#     ax.barh(age_groups, population_counts)
#     ax.set_xlabel('人口数')
#     ax.set_ylabel('年龄段')
#     ax.set_title(f'{year}年人口统计')

#     # 调整横坐标范围，预留右侧空间用于显示年份
#     ax.set_xlim(0, max(population_counts) * 1.4)

#     for i, v in enumerate(population_counts):
#         ax.text(v + 0.05, i, f"{v}", color='black', fontweight='bold')
#         ax.text(max(population_counts) * 1.2, i, year, color='gray', fontsize=15, ha='right')

#     # 保存图像为数组
#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return image


# def plot_horizontal_bar(years, population_counts, age_group):
#     """
#     绘制横向条形图展示特定年龄段的人口统计数据，并标注年龄段，返回图像数组
#     """
#     fig, ax = plt.subplots(figsize=(15, 5))  # 调整图表尺寸以适应横向条形图
#     ax.barh(years, population_counts, color='skyblue')
#     ax.set_xlabel('人口数')
#     ax.set_ylabel('年份')
#     ax.set_title(f'{age_group} 年龄段人口统计')

#     # 调整横坐标范围，预留右侧空间用于显示数值
#     ax.set_xlim(0, max(population_counts) * 1.1)

#     for i, v in enumerate(population_counts):
#         ax.text(v + 0.05, i, f"{v}", color='black', fontweight='bold', va='center')

#     # 保存图像为数组
#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return image


# def ccgenerate_video_from_single_csv(file_path):
#     all_frames = []

#     # 读取整个CSV文件并处理数据
#     df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

#     # 假设第一行是年龄段，第二行开始是各年的数据
#     years = df.columns.tolist()

#     logging.debug(f"Processing data from single CSV file for years: {years}")

#     # 如果你不想要最后一帧，则可以遍历除最后一个元素外的所有年份
#     for year in years[:-2]:  # 这将排除列表中的最后一个年份
#         age_groups = df.index.tolist()
#         population_counts = df[year].tolist()
        
#         frame = plot_vertical_bar(age_groups, population_counts, year)
#         all_frames.append(frame)

#     # 创建视频但不包括最后一年的数据
#     clip = ImageSequenceClip(all_frames, fps=1)
#     clip.write_videofile('population_video.mp4')

# def generate_video_from_single_csv(file_path):
#     all_frames = []

#     # 读取整个CSV文件并处理数据
#     df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

#     # 获取所有年份，排除最后两个年份
#     years = df.columns.tolist()  
#     years = df.rows.tolist()

#     logging.debug(f"Processing data from single CSV file for years: {years}")

#     # 使用固定的年龄段列表
#     age_groups = df.index.tolist()

#     for year in years:
#         population_counts = df[year].tolist()
        
#         frame = plot_vertical_bar(age_groups, population_counts, year)
#         all_frames.append(frame)

#     # 创建视频但不包括最后两年的数据
#     clip = ImageSequenceClip(all_frames, fps=1)
#     clip.write_videofile('population_video.mp4', codec='libx264')


# if __name__ == "__main__":
#     csv_file_path = 'Population_age_distribution_Changes_Over_Years.csv'
#     generate_video_from_single_csv(csv_file_path)