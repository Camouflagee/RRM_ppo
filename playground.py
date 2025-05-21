import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 参数设置
N = 5  # obs_dim和act_dim的维度
num_samples = 100000  # 总样本数
batch_size = 64  # 批大小
epochs = 20  # 训练轮数


# 生成训练数据
def generate_data(num_samples, N):
    # 生成观测向量（范围[-10, 10)）
    obs = torch.rand(num_samples, N) * 20 - 10

    # 生成动作向量（至少保留一个0位置）
    act_dim = torch.zeros(num_samples, N)
    for i in range(num_samples):
        num_ones = torch.randint(0, N, (1,)).item()  # 随机生成0~N-1个1
        if num_ones == N:
            num_ones = N - 1  # 保证至少一个0
        indices = torch.randperm(N)[:num_ones]
        act_dim[i, indices] = 1

    # 计算标签：未被选位置中的最大值索引
    masked_obs = obs.masked_fill(act_dim.bool(), float('-inf'))
    labels = torch.argmax(masked_obs, dim=1)

    # 组合输入：观测 + 动作状态
    inputs = torch.cat([obs, act_dim], dim=1)
    return inputs, labels


# 生成数据
inputs, labels = generate_data(num_samples, N)

# 划分训练集和验证集
split = int(0.8 * num_samples)
train_dataset = TensorDataset(inputs[:split], labels[:split])
val_dataset = TensorDataset(inputs[split:], labels[split:])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# 定义模型结构
class SequenceMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型
model = SequenceMLP(2 * N, N)  # 输入维度是obs_dim + act_dim

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss, correct = 0, 0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()

    # 验证阶段
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            val_loss += criterion(outputs, y).item()
            val_correct += (outputs.argmax(1) == y).sum().item()

    # 打印统计信息
    train_acc = correct / split
    val_acc = val_correct / (num_samples - split)
    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss / len(val_loader):.4f} | Acc: {val_acc:.4f}\n")
print('done')