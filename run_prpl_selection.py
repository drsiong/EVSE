import os
import re
import numpy as np
from subprocess import Popen, PIPE

# 定义目标域列表
targets = range(1, 16)
accuracies = []

# 循环执行每个目标域的命令
for target in targets:
    command = f"python snd_ensv_prpl.py --target {target}"
    print(f"正在运行: {command}")
    os.system(command)
    # 使用subprocess捕获输出
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # 从输出中提取准确率（假设输出中有类似"acc: 95.23"的行）
    match = re.search(r"acc:\s*([\d.]+)", stdout)
    if match:
        acc = float(match.group(1))
        accuracies.append(acc)
    else:
        print(f"⚠️ 未能在Target {target}的输出中找到准确率")

# 计算统计结果
if accuracies:
    print("\n=== 统计结果 ===")
    for i, acc in enumerate(accuracies, start=1):
        print(f"目标域 {i} 的准确率: {acc:.2f}")
    print(f"平均准确率: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
else:
    print("未获取到任何准确率数据")