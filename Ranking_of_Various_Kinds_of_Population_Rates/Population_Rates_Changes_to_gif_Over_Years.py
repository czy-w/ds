import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# 配置日志记录器
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
# 读取CSV文件数据（假设文件名为'data.csv'，且第一行为列名，第一列为指标名称）
data = pd.read_csv('Population_Rates_Changes_Over_Years.csv')

# 提取年份（假设年份列名为'Year'，且为第一列之后的列名）
years = data.columns[1:].astype(int)

# 提取数据（假设第一行为'Birth Rate'，第二行为'Death Rate'，第三行为'Natural Growth Rate'）
birth_rates = data.iloc[0, 1:].values
death_rates = data.iloc[1, 1:].values
natural_growth_rates = data.iloc[2, 1:].values

# 创建绘图及初始化线条
fig, ax = plt.subplots(figsize=(10, 6))

# 设置y轴的固定刻度值
y_ticks = [-5, 0, 5, 10, 15, 20]  # 根据实际情况调整
ax.set_yticks(y_ticks)

# 设置图形标题和坐标轴标签
ax.set_title('Population Rates Changes Over Years', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Rate (‰)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# 初始化三条线条
line_birth, = ax.plot([], [], label='Birth Rate', marker='o', color='blue', linewidth=2,
                      alpha=0.6)  # alpha用于降低非当前年份的线条透明度
line_death, = ax.plot([], [], label='Death Rate', marker='s', color='red', linewidth=2, alpha=0.6)
line_natural_growth, = ax.plot([], [], label='Natural Growth Rate', marker='^', color='green',
                               linewidth=2, alpha=0.6)

# 添加图例
ax.legend(loc='upper left', fontsize=10)

# 初始化空列表用于存储突出显示的点和文本标签
highlighted_points = []
text_labels = []

# 初始显示的年份数量（可以根据实际需求调整，先显示较少年份）
initial_years_count = 5
# 当前显示到的年份索引，初始化为初始显示的年份数量减1（索引从0开始）
current_year_index = initial_years_count - 1

# 动画更新函数
def update(frame):
    global current_year_index
    # 先判断是否还没展示完全部年份，如果没展示完则逐步增加显示的年份范围
    if frame < len(years):
        current_year_index = frame
    else:
        current_year_index = len(years) - 1

    # 更新线条的数据，只显示到当前年份的数据
    line_birth.set_data(years[:current_year_index + 1], birth_rates[:current_year_index + 1])
    line_death.set_data(years[:current_year_index + 1], death_rates[:current_year_index + 1])
    line_natural_growth.set_data(years[:current_year_index + 1], natural_growth_rates[:current_year_index + 1])

    # 清除之前的突出显示点和文本标签
    for point in highlighted_points:
        point.remove()
    for label in text_labels:
        label.set_visible(False)
    highlighted_points.clear()
    text_labels.clear()

    # 突出显示当前年份的点
    highlight_birth = ax.scatter(years[current_year_index], birth_rates[current_year_index], color='blue', s=100, zorder=5)
    highlight_death = ax.scatter(years[current_year_index], death_rates[current_year_index], color='red', s=100, zorder=5)
    highlight_natural_growth = ax.scatter(years[current_year_index], natural_growth_rates[current_year_index], color='green', s=100, zorder=5)
    highlighted_points.extend([highlight_birth, highlight_death, highlight_natural_growth])

    # 在每个折线点上标注年份（包括当前年份和其他年份，但当前年份由于突出显示可能会覆盖掉这个标签）
    for i in range(current_year_index + 1):
        text_label_birth = ax.text(years[i],
                                   birth_rates[i] + 0.5 if birth_rates[i] < y_ticks[-1] - 1 else y_ticks[-1] - 0.5,
                                   # 避免标签超出y轴范围
                                   str(years[i]), fontsize=8, ha='center',
                                   va='bottom' if birth_rates[i] < y_ticks[-1] - 1 else 'top',
                                   color='blue' if i == current_year_index else 'gray')
        text_label_death = ax.text(years[i],
                                   death_rates[i] - 0.5 if death_rates[i] > y_ticks[0] + 1 else y_ticks[0] + 0.5,
                                   # 稍微向下或向上偏移以避免重叠
                                   str(years[i]), fontsize=8, ha='center',
                                   va='top' if death_rates[i] > y_ticks[0] + 1 else 'bottom',
                                   color='red' if i == current_year_index else 'gray')
        text_label_natural_growth = ax.text(years[i],
                                            natural_growth_rates[i] + 0.25 if natural_growth_rates[i] < y_ticks[
                                                -1] - 1.25 else natural_growth_rates[i] - 0.75,  # 根据位置调整偏移量
                                            str(years[i]), fontsize=8, ha='center', va='center' if (
                                                        natural_growth_rates[i] < y_ticks[-1] - 1.25 and natural_growth_rates[i] > y_ticks[
                                                    0] + 1.25) else ('top' if natural_growth_rates[i] < y_ticks[0] + 1.25 else 'bottom'),
                                            color='green' if i == current_year_index else 'gray')
        text_labels.extend([text_label_birth, text_label_death, text_label_natural_growth])

    # 设置X轴范围只显示当前年份的数字（不显示刻度）
    ax.set_xticks([years[current_year_index]])
    ax.set_xticklabels([str(years[current_year_index])])
    ax.tick_params(axis='x', which='both', length=0)

    # y轴范围根据y_ticks设置，但确保包含所有数据点
    y_max = max(max(birth_rates[:current_year_index + 1]), max(death_rates[:current_year_index + 1]), max(natural_growth_rates[:current_year_index + 1]))
    ax.set_ylim(0, max(y_ticks[-1], y_max + (y_max // 5)))  # 稍微增加一些空间以避免截断数据点

    # 返回需要更新的线条对象（这里返回三条线条对象，确保动画能正确更新显示）
    return line_birth, line_death, line_natural_growth


# 创建动画，设置frames为年份数量，interval为帧之间的时间间隔（毫秒）
# ani = animation.FuncAnimation(fig, update, frames=len(years), interval=1000, blit=True, repeat=False)

logging.debug("开始动画")
ani = animation.FuncAnimation(fig, update, frames=len(years), interval=1000, blit=True, repeat=False)
logging.debug("结束动画")
# 如果要保存为GIF，请取消注释以下行
ani.save('Population_Rates_Changes_Over_Years.gif', writer='pillow')
logging.debug("保存动画")