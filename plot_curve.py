import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
# 设置支持中文的字体（Windows系统通常使用SimHei，Mac使用PingFang SC）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei', 'PingFang SC', 'Heiti TC']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 常见中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
x = np.arange(0, 65, 5) / 100  # 横轴数据

# 五组不同的曲线函数
y1 = np.sin(x)  # SeqPPO side info
y2 = np.cos(x)  # SeqPPO no side info
y3 = [74.444, 74.408, 74.4, 74.34, 74.254, 74.127, 73.678, 73.343, 72.51, 72.285, 71.101, 71.177, 71.259]  # MM
y4 = [73.498, 73.784, 73.532, 73.599, 73.606, 73.493, 73.13, 72.351, 72.013, 71.175, 70.473, 70.669, 70.709]  # SCA
y5 = [70.80, 70.97, 70.68, 70.69, 70.60, 70.55, 70.56, 70.49, 70.45, 70.41, 70.29, 70.16, 70.10]  # GradProj

# 创建图形
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制五组曲线
# plt.plot(x, y1, label='SeqPPO', color='blue')
# plt.plot(x, y2, label='cos(x)', color='red', linestyle='--')
plt.plot(x, y3, label='MM', color='green', linestyle=':')
plt.plot(x, y4, label='GradProj', color='purple', linestyle='-.')
plt.plot(x, y5, label='SCA', color='orange')

# 添加图例
plt.legend(loc='upper right')

# 添加轴标签
plt.xlabel('X轴名称', fontsize=12)
plt.ylabel('Y轴名称', fontsize=12)

# 添加标题
plt.title('五组曲线对比图', fontsize=14)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图形
plt.show()
