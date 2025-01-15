import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# 设置负号
plt.rcParams['axes.unicode_minus'] = False

# 设置字体
plt.rc("font", family='SimHei')

# set logging level
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)

# 配置日志记录器
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取CSV文件数据
data = pd.read_csv('Population_Rates_Changes_Over_Years.csv')

# 提取年份和数据
years = data.columns[1:].astype(int)
birth_rates = data.iloc[0, 1:].values
death_rates = data.iloc[1, 1:].values
natural_growth_rates = data.iloc[2, 1:].values

# 创建绘图及初始化线条
fig, ax = plt.subplots(figsize=(10, 6))

# 设置图形标题和坐标轴标签
ax.set_title('2004-2023中国人口出生死亡率变化统计', fontsize=16, color='white', fontweight='bold',bbox=dict(facecolor='red', edgecolor='none', pad=8.0),pad=20)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('百分率 (‰)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# 初始化三条线条
line_birth, = ax.plot([], [], label='出生率', marker='o', color='blue', linewidth=2, alpha=0.6)
line_death, = ax.plot([], [], label='死亡率', marker='s', color='red', linewidth=2, alpha=0.6)
line_natural_growth, = ax.plot([], [], label='人口自然增长率', marker='^', color='green', linewidth=2, alpha=0.6)

# 添加图例
ax.legend(loc='upper left', fontsize=10)

# 设置y轴刻度和范围
y_ticks = [-5, 0, 5, 10, 15, 20]
ax.set_yticks(y_ticks)
ax.set_ylim(-5, 20)  # 固定y轴范围从-5到20

# 初始化空列表用于存储突出显示的点和文本标签
highlighted_points = []
text_labels = []

# 初始显示的年份数量
initial_years_count = 5
current_year_index = initial_years_count - 1

plt.tight_layout()  # 自动调整子图参数以适应图表区域

# 动画更新函数
def update(frame):
    global current_year_index
    if frame < len(years):
        current_year_index = frame
    else:
        current_year_index = len(years) - 1

    line_birth.set_data(years[:current_year_index + 1], birth_rates[:current_year_index + 1])
    line_death.set_data(years[:current_year_index + 1], death_rates[:current_year_index + 1])
    line_natural_growth.set_data(years[:current_year_index + 1], natural_growth_rates[:current_year_index + 1])

    for point in highlighted_points:
        point.remove()
    for label in text_labels:
        label.set_visible(False)
    highlighted_points.clear()
    text_labels.clear()

    highlight_birth = ax.scatter(years[current_year_index], birth_rates[current_year_index], color='blue', s=100, zorder=5)
    highlight_death = ax.scatter(years[current_year_index], death_rates[current_year_index], color='red', s=100, zorder=5)
    highlight_natural_growth = ax.scatter(years[current_year_index], natural_growth_rates[current_year_index], color='green', s=100, zorder=5)
    highlighted_points.extend([highlight_birth, highlight_death, highlight_natural_growth])

    for i in range(current_year_index + 1):
        text_label_birth = ax.text(years[i], birth_rates[i] + 0.5 if birth_rates[i] < 19 else 19, str(years[i]), fontsize=8, ha='center', va='bottom' if birth_rates[i] < 19 else 'top', color='blue' if i == current_year_index else 'gray')
        text_label_death = ax.text(years[i], death_rates[i] - 0.5 if death_rates[i] > -4 else -4, str(years[i]), fontsize=8, ha='center', va='top' if death_rates[i] > -4 else 'bottom', color='red' if i == current_year_index else 'gray')
        text_label_natural_growth = ax.text(years[i], natural_growth_rates[i] + 0.25 if natural_growth_rates[i] < 19 else natural_growth_rates[i] - 0.75, str(years[i]), fontsize=8, ha='center', va='center' if natural_growth_rates[i] < 19 and natural_growth_rates[i] > -4 else ('top' if natural_growth_rates[i] < -4 else 'bottom'), color='green' if i == current_year_index else 'gray')
        text_labels.extend([text_label_birth, text_label_death, text_label_natural_growth])

    ax.set_xticks([years[current_year_index]])
    ax.set_xticklabels([str(years[current_year_index])])
    ax.tick_params(axis='x', which='both', length=0)

    return line_birth, line_death, line_natural_growth


def extended_frames():
    for frame in range(len(years)):
        for _ in range(4):  # 每个原始帧重复3次
            yield frame

# 创建动画对象ani，使用extended_frames作为帧源
logging.debug("开始动画")
ani = animation.FuncAnimation(fig, update, frames=extended_frames, interval=1000, blit=True, repeat=False)
logging.debug("结束动画")

# 保存为MP4视频
ani.save('Population_Rates_Changes_Over_Years.mp4', writer='ffmpeg', fps=1)
logging.debug("保存动画")

# 显示动画
# plt.show()