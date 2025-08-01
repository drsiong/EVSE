# -*- encoding: utf-8 -*-
# 五个候选模型绘图
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = '10'
matplotlib.rcParams['pgf.texsystem'] = 'xelatex'
matplotlib.rcParams['pgf.preamble'] = r'\usepackage{XeCJK}'
matplotlib.rcParams['legend.frameon'] = 'True'
# 数据
methods = ['origin', '0.01', '0.05', '0.1', '0.15', '0.2']
accuracy = [88.31, 87.50, 88.85, 89.09, 88.83, 87.96]  # 示例数据
error = [5.73, 6.24, 5.68, 5.27, 5.35, 5.67]  # 误差范围

# 定义颜色和样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 6种颜色
markers = ['o', 's', 'D', '^', 'v', 'p']  # 6种标记样式
sizes = [80, 100, 120, 150, 120, 100]  # 不同点的大小

# 调整 X 轴位置
x_pos = np.arange(len(methods)) # 点的位置

# 创建图表
plt.figure(figsize=(10, 6))

plt.plot(x_pos, accuracy, marker='o', color='black', linestyle='-', markersize=8, label='Accuracy') 
# 绘制点图并连线
for i in range(len(methods)):
    plt.plot(x_pos[i], accuracy[i], marker=markers[i], color=colors[i], linestyle='-', markersize=np.sqrt(sizes[i]), label=methods[i])  # 绘制点和折线
    plt.errorbar(x_pos[i], accuracy[i], yerr=error[i], fmt='none', color=colors[i], capsize=5, elinewidth=2)  # 误差条

# 找到最高准确率的索引
max_index = np.argmax(accuracy)

# 绘制最高准确率的点（加粗）
plt.plot(x_pos[max_index], accuracy[max_index], marker=markers[max_index], color=colors[max_index], linestyle='none', markersize=np.sqrt(sizes[max_index]), markeredgecolor='black', markeredgewidth=2, zorder=5)  # 加粗显示

# 添加数据点标签
for i in range(len(methods)):
    plt.text(x_pos[i], accuracy[i] + 0.5, f'{accuracy[i]:.2f}', color=colors[i], ha='center', va='bottom', fontsize=12, fontweight='bold')

# 突出 0.1 的效果
plt.axvspan(x_pos[max_index] - 0.5, x_pos[max_index] + 0.5, color=colors[max_index], alpha=0.1)  # 添加背景色
plt.text(x_pos[max_index], accuracy[max_index] - 2, 'Best Performance', color=colors[max_index], ha='center', va='top', fontsize=14, fontweight='bold')  # 添加注释

# 添加标题和标签
plt.xlabel('Different Flu_level', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy(%)', fontsize=14, fontweight='bold')
plt.title('Accuracy with Different Flu_level', fontsize=16, fontweight='bold')
plt.xticks(x_pos, methods, rotation=45, fontsize=12)  # 旋转 X 轴标签
plt.ylim(80, 96)  # 纵轴从 65 开始

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 显示图表
plt.tight_layout()
# 在 plt.show() 之前添加这行代码：
plt.savefig('dflu_accuracy.pdf')  
plt.show()