import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
from result_process_api import record_event

# 配置日志记录器
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

# plt.ion()

data = pd.read_csv('Population_age_distribution_Changes_Over_Years.csv')
years = data.columns[1:].astype(int)

num_0_4 = data.iloc[0, 1:].values
num_5_9 = data.iloc[1, 1:].values
num_10_14 = data.iloc[2, 1:].values
num_15_19 = data.iloc[3, 1:].values
num_20_24 = data.iloc[4, 1:].values
num_25_29 = data.iloc[5, 1:].values
num_30_34 = data.iloc[6, 1:].values
num_35_39 = data.iloc[7, 1:].values
num_40_44 = data.iloc[8, 1:].values
num_45_49 = data.iloc[9, 1:].values
num_50_54 = data.iloc[10, 1:].values
num_55_59 = data.iloc[11, 1:].values
num_60_64 = data.iloc[12, 1:].values
num_65_69 = data.iloc[13, 1:].values
num_70_74 = data.iloc[14, 1:].values
num_75_79 = data.iloc[15, 1:].values
num_80_84 = data.iloc[16, 1:].values
num_85_89 = data.iloc[17, 1:].values
num_90_94 = data.iloc[18, 1:].values
num_95_ = data.iloc[19, 1:].values
num_all = data.iloc[20, 1:].values

logging.debug(f"坐标系{len(years)}帧数")
logging.debug("开始初始化坐标系")
# 创建绘图及初始化线条fig为图，ax为坐标系统，设置图形标题和坐标轴标签,设置坐标轴范围
fig, ax = plt.subplots(figsize=(100, 100))
y_ticks = [-200000, 0, 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000]
ax.set_title('Population age distribution Changes Over Years', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Num (millon people)', fontsize=12)
# ax.set_xlim()
# ax.set_ylim()
ax.tick_params(axis='both', which='major', labelsize=10)  #设置坐标轴刻度（ticks）的属性的函数


# 初始化线条
line_num_0_4, = ax.plot([], [], label='num 0-4', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_5_9, = ax.plot([], [], label='num 5-9', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_10_14, = ax.plot([], [], label='num 10-14', marker='o', color='blue', linewidth=2, alpha=0.6)
line_num_15_19, = ax.plot([], [], label='num 15-19', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_20_24, = ax.plot([], [], label='num 20-24', marker='o', color='blue', linewidth=2, alpha=0.6)
line_num_25_29, = ax.plot([], [], label='num 25-29', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_30_34, = ax.plot([], [], label='num 30-34', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_35_39, = ax.plot([], [], label='num 35-39', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_40_44, = ax.plot([], [], label='num 40-44', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_45_49, = ax.plot([], [], label='num 45-49', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_50_54, = ax.plot([], [], label='num 50-54', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_55_59, = ax.plot([], [], label='num 55-59', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_60_64, = ax.plot([], [], label='num 60-64', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_65_69, = ax.plot([], [], label='num 65-69', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_70_74, = ax.plot([], [], label='num 70-74', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_75_79, = ax.plot([], [], label='num 75-79', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_80_84, = ax.plot([], [], label='num 80-84', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_85_89, = ax.plot([], [], label='num 85-89', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_90_94, = ax.plot([], [], label='num 90-94', marker='s', color='red', linewidth=2, alpha=0.6)
line_num_95_, = ax.plot([], [], label='num 95-', marker='^', color='green', linewidth=2, alpha=0.6)
line_num_all, = ax.plot([], [], label='num all', marker='^', color='green', linewidth=2, alpha=0.6)

ax.legend(loc='upper left', fontsize=10)
highlighted_points = []
text_labels = []
initial_years_count = 5
current_year_index = initial_years_count - 1
logging.debug("结束初始化坐标系")


@record_event
def update(frame):
    global current_year_index
    logging.debug(f"坐标系有{frame}帧")
    # current_year_index = frame
    # 先判断是否还没展示完全部年份，如果没展示完则逐步增加显示的年份范围
    if frame < len(years):
        current_year_index = frame
    else:
        current_year_index = len(years) - 1

    # 更新线条的数据，只显示到当前年份的数据
    logging.debug(f"更新线条的数据{years[:current_year_index + 1],num_0_4[:current_year_index + 1]}")
    line_num_0_4.set_data(years[:current_year_index + 1], num_0_4[:current_year_index + 1])
    line_num_5_9.set_data(years[:current_year_index + 1], num_5_9[:current_year_index + 1])
    line_num_10_14.set_data(years[:current_year_index + 1], num_10_14[:current_year_index + 1])
    line_num_15_19.set_data(years[:current_year_index + 1], num_15_19[:current_year_index + 1])
    line_num_20_24.set_data(years[:current_year_index + 1], num_20_24[:current_year_index + 1])
    line_num_25_29.set_data(years[:current_year_index + 1], num_25_29[:current_year_index + 1])    
    line_num_30_34.set_data(years[:current_year_index + 1], num_30_34[:current_year_index + 1])
    line_num_35_39.set_data(years[:current_year_index + 1], num_35_39[:current_year_index + 1])
    line_num_40_44.set_data(years[:current_year_index + 1], num_40_44[:current_year_index + 1])
    line_num_45_49.set_data(years[:current_year_index + 1], num_45_49[:current_year_index + 1])
    line_num_50_54.set_data(years[:current_year_index + 1], num_50_54[:current_year_index + 1])
    line_num_55_59.set_data(years[:current_year_index + 1], num_55_59[:current_year_index + 1])    
    line_num_60_64.set_data(years[:current_year_index + 1], num_60_64[:current_year_index + 1])
    line_num_65_69.set_data(years[:current_year_index + 1], num_65_69[:current_year_index + 1])
    line_num_70_74.set_data(years[:current_year_index + 1], num_70_74[:current_year_index + 1])
    line_num_75_79.set_data(years[:current_year_index + 1], num_75_79[:current_year_index + 1])
    line_num_80_84.set_data(years[:current_year_index + 1], num_80_84[:current_year_index + 1])
    line_num_85_89.set_data(years[:current_year_index + 1], num_85_89[:current_year_index + 1])
    line_num_90_94.set_data(years[:current_year_index + 1], num_90_94[:current_year_index + 1])
    line_num_95_.set_data(years[:current_year_index + 1], num_95_[:current_year_index + 1])
    line_num_all.set_data(years[:current_year_index + 1], num_all[:current_year_index + 1])
    logging.debug("更新线条的数据结束一轮")


    # 清除之前的突出显示点和文本标签
    for point in highlighted_points:
        point.remove()
    for label in text_labels:
        label.set_visible(False)
    highlighted_points.clear()
    text_labels.clear()

    # # # 突出显示当前年份的点
    logging.debug(f"突出显示当前年份的点{years[current_year_index],num_0_4[current_year_index]}")
    highlight_num_0_4 = ax.scatter(years[current_year_index], num_0_4[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_5_9 = ax.scatter(years[current_year_index], num_5_9[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_10_14 = ax.scatter(years[current_year_index], num_10_14[current_year_index], color='red', s=100, zorder=5)
    highlight_num_15_19 = ax.scatter(years[current_year_index], num_15_19[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_20_24 = ax.scatter(years[current_year_index], num_20_24[current_year_index], color='green', s=100, zorder=5)
    highlight_num_25_29 = ax.scatter(years[current_year_index], num_25_29[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_30_34 = ax.scatter(years[current_year_index], num_30_34[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_35_39 = ax.scatter(years[current_year_index], num_35_39[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_40_44 = ax.scatter(years[current_year_index], num_40_44[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_45_49 = ax.scatter(years[current_year_index], num_45_49[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_50_54 = ax.scatter(years[current_year_index], num_50_54[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_55_59 = ax.scatter(years[current_year_index], num_55_59[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_60_64 = ax.scatter(years[current_year_index], num_60_64[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_65_69 = ax.scatter(years[current_year_index], num_65_69[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_70_74 = ax.scatter(years[current_year_index], num_70_74[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_75_79 = ax.scatter(years[current_year_index], num_75_79[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_80_84 = ax.scatter(years[current_year_index], num_80_84[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_85_89 = ax.scatter(years[current_year_index], num_85_89[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_90_94 = ax.scatter(years[current_year_index], num_90_94[current_year_index], color='blue', s=100, zorder=5)
    highlight_num_95_ = ax.scatter(years[current_year_index], num_95_[current_year_index], color='blue', s=100, zorder=5)                          
    highlight_num_all = ax.scatter(years[current_year_index], num_all[current_year_index], color='blue', s=100, zorder=5)                          
    highlighted_points.extend([highlight_num_0_4, highlight_num_5_9, highlight_num_10_14, highlight_num_15_19,highlight_num_20_24,
                            highlight_num_25_29,highlight_num_30_34,highlight_num_35_39,highlight_num_40_44,highlight_num_45_49,
                            highlight_num_50_54,highlight_num_55_59,highlight_num_60_64,highlight_num_65_69,highlight_num_70_74,
                            highlight_num_75_79,highlight_num_80_84,highlight_num_85_89,highlight_num_90_94,
                            highlight_num_95_, highlight_num_all])
    logging.debug("突出显示当前年份的点一轮结束")
    

    # 在每个折线点上标注年份（包括当前年份和其他年份，但当前年份由于突出显示可能会覆盖掉这个标签）
    for i in range(current_year_index + 1):
        logging.debug(f"标注年份{years[i],num_0_4[i]}")
        text_num_0_4 = ax.text(years[i], num_0_4[i] + 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='blue')
        text_num_5_9 = ax.text(years[i], num_5_9[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='red')
        text_num_10_14 = ax.text(years[i], num_10_14[i] + 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='green')
        text_num_15_19 = ax.text(years[i], num_15_19[i] - 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='purple')
        text_num_20_24 = ax.text(years[i], num_20_24[i] + 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='orange')
        text_num_25_29 = ax.text(years[i], num_25_29[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='pink')
        text_num_30_34 = ax.text(years[i], num_30_34[i] + 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='brown')
        text_num_35_39 = ax.text(years[i], num_35_39[i] - 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='gray')
        text_num_40_44 = ax.text(years[i], num_40_44[i] + 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='cyan')
        text_num_45_49 = ax.text(years[i], num_45_49[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='magenta')
        text_num_50_54 = ax.text(years[i], num_50_54[i] + 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='lime')
        text_num_55_59 = ax.text(years[i], num_55_59[i] - 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='navy')
        text_num_60_64 = ax.text(years[i], num_60_64[i] + 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='gold')
        text_num_65_69 = ax.text(years[i], num_65_69[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='teal')
        text_num_70_74 = ax.text(years[i], num_70_74[i] + 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='maroon')
        text_num_75_79 = ax.text(years[i], num_75_79[i] - 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='silver')
        text_num_80_84 = ax.text(years[i], num_80_84[i] + 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='olive')
        text_num_85_89 = ax.text(years[i], num_85_89[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='indigo')
        text_num_90_94 = ax.text(years[i], num_90_94[i] + 0.25, str(years[i]), fontsize=8, ha='center', va='center', color='violet')
        text_num_95_ = ax.text(years[i], num_95_[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='red')
        text_num_all = ax.text(years[i], num_all[i] - 0.5, str(years[i]), fontsize=8, ha='center', va='top', color='red')

        text_labels.extend([text_num_0_4, text_num_5_9, text_num_10_14, text_num_15_19, text_num_20_24, 
                            text_num_25_29, text_num_30_34, text_num_35_39, text_num_40_44, text_num_45_49, 
                            text_num_50_54, text_num_55_59, text_num_60_64, text_num_65_69, text_num_70_74, 
                            text_num_75_79, text_num_80_84, text_num_85_89, text_num_90_94, text_num_95_, text_num_all])
        logging.debug("标注年份结束一轮")

    # 设置X轴范围只显示当前年份的数字（不显示刻度）
    logging.debug("X轴范围只显示当前年份")
    ax.set_xticks([years[current_year_index]])
    ax.set_xticklabels([str(years[current_year_index])])
    ax.tick_params(axis='x', which='both', length=0)
    logging.debug("X轴范围只显示当前年份结束")

    # y轴范围根据y_ticks设置，但确保包含所有数据点
    # y_max = max(max(num_0_4[:current_year_index + 1]), max(num_10_14[:current_year_index + 1]), max(num_20_24[:current_year_index + 1]))
    # ax.set_ylim(0, max(y_ticks[-1], y_max + (y_max // 5)))  # 稍微增加一些空间以避免截断数据点
    logging.debug("结束动画一轮")

    # 返回需要更新的线条对象（这里返回三条线条对象，确保动画能正确更新显示）
    ne =  [line_num_0_4, line_num_5_9, line_num_10_14, line_num_15_19, line_num_20_24, line_num_25_29, line_num_30_34, line_num_35_39, line_num_40_44, line_num_45_49, line_num_50_54, line_num_55_59, line_num_60_64, line_num_65_69, line_num_70_74, line_num_75_79, line_num_80_84, line_num_85_89, line_num_90_94, line_num_95_, text_num_all]
    return [highlighted_points, text_labels, ne]

logging.debug("开始动画")
ani = animation.FuncAnimation(fig, update, frames=len(years), interval=500, blit=False, repeat=False)
logging.debug("结束动画")
# 如果要保存为GIF，请取消注释以下行
ani.save('Population_age_distribution_Changes_Over_Years.gif', writer='pillow')
logging.debug("保存动画")