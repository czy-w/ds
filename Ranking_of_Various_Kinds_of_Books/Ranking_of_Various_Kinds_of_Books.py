# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from moviepy.editor import ImageSequenceClip
import os
import logging
import matplotlib

# set fonts
matplotlib.rc("font",family='SimHei')

# set logging level
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)

# set logger
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')


def read_csv_and_process(file_path):
    """
    读取CSV文件并返回处理后的数据（电影名和票房数据以及年份）
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df.head(10)
    movie_names = [name.replace('\u200c', '') for name in df.iloc[:, 1].tolist()]
    box_office = [int(box_office) for box_office in df.iloc[:, 3].tolist()]
    year = os.path.splitext(os.path.basename(file_path))[0]
    logging.debug(f"Processing data for year: {movie_names, box_office, year}")
    # return movie_names, box_office, year
    return list(zip(movie_names, box_office, [year]*len(movie_names)))


def collect_all_movies_data():
    """
    收集所有年份的电影票房数据，并返回一个包含所有电影及其票房和年份的列表。
    """
    all_movies = []
    file_paths = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
    file_paths.sort()

    for file_path in file_paths:
        all_movies.extend(read_csv_and_process(file_path))
    
    # 按票房从高到低排序
    all_movies.sort(key=lambda x: x[1], reverse=True)
    return all_movies


def plot_horizontal_bar(movie_names, box_office, year, fig_size=(38,38)):
    """
    绘制横向条形图展示电影票房数据，并标注年份，返回图像数组
    """
    plt.rcParams.update({'font.size': 30})  # 调整此值以适应您的需要
    # 将票房和电影名配对并根据票房降序排序
    sorted_data = sorted(zip(box_office, movie_names), key=lambda x: x[0], reverse=True)
    box_office_sorted, movie_names_sorted = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # 使用range生成y轴刻度位置，并反转以使最高票房位于顶部
    positions = range(len(movie_names_sorted))
    ax.barh(positions, box_office_sorted, color=[plt.cm.viridis(i/len(movie_names_sorted)) for i in range(len(movie_names_sorted))])

    # 隐藏y轴刻度线和标签
    # ax.yaxis.set_visible(False)    
    
    # 设置y轴标签为电影名，并反转y轴顺序使最高票房位于顶部
    ax.set_yticks(positions)
    ax.set_yticklabels(movie_names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel('阅读人数', fontsize=30, color='gray')
    ax.set_ylabel('书名', fontsize=30, color='gray')
    ax.set_title(f'{year}年综评前10书籍', fontsize=60, color='white', fontweight='bold',bbox=dict(facecolor='red', edgecolor='red', boxstyle='round,pad=0.5'),x=0.4)


    # 调整横坐标范围，预留右侧空间用于显示年份
    max_box_office = max(box_office_sorted) if box_office_sorted else 1  # 防止空列表导致的错误
    ax.set_xlim(-max_box_office * 0.1, max_box_office * 1.4)  # 在左侧预留更多空间


    for i, v in enumerate(box_office_sorted):
        ax.text(v + 0.05, i, f"{v}", color='green', fontweight='bold', va='center')
        rank = i + 1  # 计算排名，从1开始
        ax.text(max(box_office_sorted) * 1.2, i, f"#{rank}", color='blue', fontsize=30, ha='right', va='center')


    plt.subplots_adjust(left=0.3)  

    return save_and_close(fig)


def plot_top_50_overall(all_movies, fig_size=(38,38)):
    """
    绘制所有年份中票房前50的电影汇总图表，并返回图像数组
    """

    plt.rcParams.update({'font.size': 30})  # 调整此值以适应您的需要
    # 获取并根据票房降序排序前50的电影
    top_50_movies_sorted = sorted(all_movies, key=lambda x: x[1], reverse=True)[:50]

    fig, ax = plt.subplots(figsize=fig_size)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    movie_names, box_office, years = zip(*top_50_movies_sorted)

    # 使用range(len(movie_names))生成y轴刻度位置，并反转以使最高票房位于顶部
    positions = range(len(movie_names))
    ax.barh(positions, box_office, color=[plt.cm.viridis(i/len(top_50_movies_sorted)) for i in range(len(top_50_movies_sorted))])
    
    # 设置y轴标签为电影名，并反转y轴顺序使最高票房位于顶部
    ax.set_yticks(positions)
    ax.set_yticklabels(movie_names)
    ax.invert_yaxis()  # 反转y轴使得最高的条形在顶部

    ax.set_xlabel('阅读人数', fontsize=30, color='gray') #,position=(0, 0.5))
    ax.set_ylabel('书名', fontsize=30, color='gray') #,position=(0, 0.5))
    ax.set_title('2011-2024年综评前50书籍汇总', fontsize=60, color='white', fontweight='bold',bbox=dict(facecolor='red', edgecolor='red', boxstyle='round,pad=0.5'),x=0.4)

    # 添加票房数值标签和年份信息
    sorted_indices = sorted(range(len(box_office)), key=lambda i: box_office[i], reverse=True)
    rank_dict = {i: idx+1 for idx, i in enumerate(sorted_indices)}

    for i, v in enumerate(box_office):
        # 使用rank_dict根据原始索引来查找排名。
        rank = rank_dict[i]
        ax.text(v + 0.05, i, f"{v} ({years[i]})",color='green', fontsize=30, fontweight='bold', va='center')
        ax.text(max(box_office) * 1.3, i, f"#{rank}",color='blue', fontsize=30, fontweight='bold', va='center', ha='right')

    # 调整横坐标范围
    max_box_office = max(box_office) if box_office else 1  # 防止空列表导致的错误
    ax.set_xlim(-max_box_office * 0.1, max_box_office * 1.3)  # 在左侧预留更多空间

    # 调整图表布局，为左侧文字留出更多空间
    plt.subplots_adjust(left=0.2)  # 调整这个值可以改变左侧的空间大小

    return save_and_close(fig)

def save_and_close(fig):
    """
    保存图像为数组并关闭图表，确保所有图表尺寸一致
    """
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def generate_video_with_top_50_overall():
    all_frames = []
    file_paths = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
    file_paths.sort()

    # 增加每一帧的停留时间
    for file_path in file_paths:
        movies_data = read_csv_and_process(file_path)[:10]
        movie_names, box_office, year = zip(*movies_data)
        frame = plot_horizontal_bar(movie_names, box_office, year[0])
        all_frames.extend([frame]*7)  # 每帧重复3次以延长显示时间

    # 添加所有年份前50名的汇总图表帧
    all_movies = collect_all_movies_data()
    summary_frame = plot_top_50_overall(all_movies)
    all_frames.extend([summary_frame]*14)  # 汇总帧重复5次以延长显示时间

    clip = ImageSequenceClip(all_frames, fps=1)  # 减少FPS以延长每一帧的显示时间
    clip.write_videofile('Ranking_of_Various_Kinds_of_Books.mp4', codec='libx264')


if __name__ == "__main__":
    generate_video_with_top_50_overall()