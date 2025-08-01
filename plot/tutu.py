import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = '10'
matplotlib.rcParams['pgf.texsystem'] = 'xelatex'
matplotlib.rcParams['pgf.preamble'] = r'\usepackage{XeCJK}'
matplotlib.rcParams['legend.frameon'] = 'True'

# 准备数据
data = {
    'target': [f'target{i}' for i in range(1, 16)],
    '0.01': [86.89, 80.47, 83.94, 92.34, 83.56, 77.81, 88.95, 86.48, 85.15, 85.71, 97.44, 84.89, 77.70, 87.89, 80.47],
    '0.1': [91.57, 82.59, 82.79, 92.66, 84.47, 81.53, 88.95, 85.30, 79.79, 89.63, 99.12, 82.17, 79.26, 82.23, 85.47],
    '0.5': [90.13, 83.00, 84.15, 98.79, 87.01, 83.91, 89.42, 85.89, 87.74, 87.07, 99.06, 87.45, 78.05, 86.30, 86.09],
    '1': [92.55, 84.77, 85.74, 100.00, 84.12, 85.92, 85.44, 83.71, 92.87, 87.63, 97.70, 83.68, 77.70, 92.72, 90.13],
    '2': [92.52, 88.01, 77.96, 93.05, 84.41, 83.71, 88.45, 85.56, 88.48, 89.95, 94.70, 86.39, 81.70, 85.50, 80.64],
    'selected': [92.52, 88.01, 85.74, 100.00, 84.47, 83.71, 89.42, 85.89, 92.87, 89.95, 99.06, 84.89, 81.70, 92.72, 85.47]
}

df = pd.DataFrame(data)

# 选择5个目标域
selected_targets = ['target2', 'target3', 'target7', 'target13', 'target14']
df = df[df['target'].isin(selected_targets)].reset_index(drop=True)  # 重置索引

# 设置绘图风格
# plt.style.use('seaborn-v0_8-pastel')
plt.figure(figsize=(10, 6), dpi=100)

# 定义颜色和线型
colors = {
    '0.01': '#1f77b4',  # 蓝色
    '0.1': '#ff7f0e',   # 橙色
    '0.5': '#2ca02c',   # 绿色
    '1': '#d62728',     # 红色
    '2': '#9467bd',     # 紫色
    'selected': '#e377c2'  # 粉色
}
line_styles = ['-', '--', '-.', ':', '-', '-']

# 绘制每条曲线
for i, col in enumerate(['0.01', '0.1', '0.5', '1', '2', 'selected']):
    plt.plot(df['target'], df[col], 
             label=f'model={col}' if col != 'selected' else 'Selected Model',
             color=colors[col],
             linestyle=line_styles[i],
             linewidth=2 if col == 'selected' else 1.5,
             marker='o' if col == 'selected' else None,
             markersize=6 if col == 'selected' else 0)

# 标注所有 selected 数据点（不再检查是否为最大值）
for idx, row in df.iterrows():
    plt.annotate(f'{row["selected"]:.2f}',  # 保留2位小数
                 xy=(idx, row['selected']),
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 color=colors['selected'],
                 weight='bold',
                 fontsize=9)  # 调整字体大小避免重叠

# 图表美化
plt.title('Model Accuracy Across Selected Targets', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Target', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(75, 102)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整布局
plt.tight_layout()

# 保存或显示
plt.savefig('accuracy_plot_selected.pdf')
plt.show()