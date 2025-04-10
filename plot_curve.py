import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
# 设置支持中文的字体（Windows系统通常使用SimHei，Mac使用PingFang SC）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei', 'PingFang SC', 'Heiti TC']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 常见中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
x = np.arange(0, 55, 5)  # 横轴数据

# 五组不同的曲线函数
y1 = [85.74399016531734, 89.39694075272786, 83.35550990879818, 90.2886813804711, 87.79312515256845, 82.4841580372947, 88.97286711099093, 85.01144788785767, 87.5544473749781, 84.18119389522987, 81.25782494381957, 81.08866814105485, 75.11656224695608]  # SeqPPO side info
y2 = [83.06747609543748, 81.21936063269168, 84.33196743263397, 86.37340354357701, 81.32640719853036, 86.09375537148375, 85.78459035181882, 82.9172677636367, 81.9780241872672, 82.65802392319897, 81.28525346053699, 83.7535748951151, 82.95794848620707]
y3 = [74.444, 74.408, 74.4, 74.34, 74.254, 74.127, 73.678, 73.343, 72.51, 72.285, 71.101, 71.177, 71.259]  # MM
y4 = [73.498, 73.784, 73.532, 73.599, 73.606, 73.493, 73.13, 72.351, 72.013, 71.175, 70.473, 70.669, 70.709]  # SCA
y5 = [70.80, 70.97, 70.68, 70.69, 70.60, 70.55, 70.56, 70.49, 70.45, 70.41, 70.29, 70.16, 70.10]  # GradProj

# 创建图形
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制五组曲线
plt.plot(x, y1[:len(x)], label='SeqPPO-SI', color='blue')
plt.plot(x, y2[:len(x)], label='SeqPPO-NoSI', color='red', linestyle='--')
plt.plot(x, y3[:len(x)], label='MM', color='green', linestyle=':')
plt.plot(x, y4[:len(x)], label='GradProj', color='purple', linestyle='-.')
plt.plot(x, y5[:len(x)], label='SCA', color='orange')

# 设置横轴刻度
plt.xticks(x)
# 添加图例
plt.legend(loc='upper right')

# 添加轴标签
plt.xlabel('估计误差百分比', fontsize=12)
plt.ylabel('系统总和速率(Mbps)', fontsize=12)

# 添加标题
plt.title('在信道有估计误差时的性能表现', fontsize=14)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图形
plt.show()

