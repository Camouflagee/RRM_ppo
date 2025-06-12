import numpy as np
import yaml
from torch import nn
import torch

from utils import DotDic, load_env

def generate_problem_instance(init_env, env_args):
    """
    生成无线通信系统的信道状态信息

    返回:
        H: 信道增益矩阵 (UE x RB)
        P: 发射功率矩阵 (UE x RB)
        n0: 噪声功率 (标量)
        N_rb: 每个用户的最大资源块数 (标量)
    """

    # 选择特定场景 (UE=12, RB=30)
    nUE, nRB = 12, 30
    # 加载环境配置

    env = init_env

    # 获取系统参数
    K = env.nRB  # 资源块数
    U = env.nUE  # 用户数
    # N_rb = nRB // 2  # 每个用户的最大资源块数

    # 重置环境获取初始状态
    obs, info = env.reset_onlyforbaseline()

    # 获取发射功率 (所有用户和资源块使用相同功率)
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((U, K)) * P_constant

    # 获取信道状态信息 (CSI) 并转换为线性值
    H_uk = 10 ** (info['CSI'] / 10)  #从dBm转换为线性值
    H = H_uk.reshape(U,K)
    # 获取噪声功率
    n0 = env.get_n0()

    return H, P, n0


import torch
import torch.nn as nn

import numpy as np
from utils import DotDic  # 假设DotDic在你的环境中可用


class UnrolledResourceAllocator(nn.Module):
    def __init__(self, K_layers, nUE, nRB, init_eta=0.1, eps=1e-8):
        """
        资源分配展开网络

        Args:
            K_layers (int): 网络层数（即迭代次数）
            nUE (int): 用户数量
            nRB (int): 资源块数量
            init_eta (float): 初始学习率值
            eps (float): 数值稳定性常数
        """
        super().__init__()
        self.K_layers = K_layers
        self.nUE = nUE
        self.nRB = nRB
        self.eps = eps

        # 为每一层创建可训练的学习率参数
        self.etas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_eta),requires_grad=True) for _ in range(K_layers)
        ])

    def forward(self, a_init, H, P, n0):
        """
        前向传播

        Args:
            a_init (torch.Tensor): 初始分配矩阵 [batch, UE, RB]
            H (torch.Tensor): 信道增益矩阵 [batch, UE, RB]
            P (torch.Tensor): 功率矩阵 [batch, UE, RB]
            n0 (float): 噪声功率

        Returns:
            torch.Tensor: 最终分配矩阵 [batch, UE, RB]
        """
        # 初始化分配矩阵
        a = a_init.clone()
        batch_size = a.size(0)

        # 确保n0正确广播 - 修正点
        if n0.dim() == 0:
            # 标量 -> [1, 1, 1]
            n0_tensor = n0.view(1, 1, 1).expand(batch_size, 1, 1)
        elif n0.dim() == 1:
            # [batch] -> [batch, 1, 1]
            n0_tensor = n0.view(batch_size, 1, 1)
        else:
            # 已经是 [batch, 1, 1] 或兼容形状
            n0_tensor = n0

        # 迭代K次（网络层数）
        for t in range(self.K_layers):
            # 步骤1：更新辅助变量w
            # 计算干扰+噪声：I = n0 + Σ_{u'≠u} a_{k,u'} * P_{k,u'} * |H_{k,u'}|^2
            # 注意: 广播用于批次处理 [batch, UE, RB] -> [batch, 1, RB]
            interference = torch.sum(a * P * H, dim=1, keepdim=True)  # 总干扰
            I = n0_tensor  + interference - a * P * H  # 排除当前用户的自身干扰

            # 计算辅助变量 w = sqrt(a * P * H) / I
            w = torch.sqrt(a * P * H + self.eps) / (I + self.eps)

            # 步骤2：更新分配矩阵a
            # 计算中间变量 s
            s = 2 * w * torch.sqrt(a * P * H + self.eps) - w.pow(2) * I

            # 计算梯度直接项
            g_direct = w * torch.sqrt(P * H) / ((1 + s) * torch.sqrt(a + self.eps))

            # 计算梯度干扰项
            # 计算辅助矩阵 Q = w^2 * P / (1 + s)
            Q = w.pow(2) * P / (1 + s + self.eps)

            # 对所有用户求和（同一资源块上）
            Q_sum = torch.sum(Q, dim=1, keepdim=True)  # [batch, 1, RB]

            # 计算干扰项: g_interf = -H * (Q_sum - Q)
            g_interf = -H * (Q_sum - Q)

            # 完整梯度 = 直接项 + 干扰项
            g = g_direct + g_interf

            # 梯度上升更新: a = a + eta * gradient
            a = a + self.etas[t] * g

            # 可选的: 数值稳定性处理（防止溢出）
            a = torch.clamp(a, 0.0, 1.0)

        return a

def generate_batch_problem_instance(init_env,env_args, batch_size=32):
    """
    批量生成问题实例

    Returns:
        H: 信道增益矩阵 [batch, UE, RB]
        P: 发射功率矩阵 [batch, UE, RB]
        n0: 噪声功率 [batch] 或 标量
    """
    # （此处保持原始generate_problem_instance函数不变）
    # 在实际实现中需要修改为支持批量生成
    # 以下是伪代码表示:
    batch_H = []
    batch_P = []
    batch_n0 = []

    for _ in range(batch_size):
        H, P, n0 = generate_problem_instance(init_env, env_args)  # 调用原始单实例函数
        batch_H.append(H)
        batch_P.append(P)
        batch_n0.append(n0)

    return np.array(batch_H), np.array(batch_P), np.array(batch_n0)


# 训练配置
nUE = 12
nRB = 30
K_layers = 10  # 展开层数
n_epochs = 0
batch_size = 1500
lr = 1e-4

# 初始化网络
model = UnrolledResourceAllocator(
    K_layers=K_layers,
    nUE=nUE,
    nRB=nRB,
    init_eta=0.1
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))

# 加载预训练环境
init_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')
# 训练循环
for epoch in range(n_epochs):
    # 生成批量问题实例
    H_np, P_np, n0_np = generate_batch_problem_instance(init_env, env_args, batch_size)

    # 转换为PyTorch张量
    H = torch.tensor(H_np, dtype=torch.float32)
    P = torch.tensor(P_np, dtype=torch.float32)
    n0 = torch.tensor(n0_np, dtype=torch.float32)

    # 初始化a（均匀分配）
    a_init = torch.full((batch_size, nUE, nRB), 0.5)

    # 前向传播
    a_out = model(a_init, H, P, n0)

    # 确保n0正确广播 - 修正点
    if n0.dim() == 0:
        # 标量 -> [1, 1, 1]
        n0_tensor = n0.view(1, 1, 1).expand(batch_size, 1, 1)
    elif n0.dim() == 1:
        # [batch] -> [batch, 1, 1]
        n0_tensor = n0.view(batch_size, 1, 1)
    else:
        # 已经是 [batch, 1, 1] 或兼容形状
        n0_tensor = n0
    # 计算目标函数（最大化总速率）
    # with torch.no_grad():
    # 计算信干噪比
    interference = torch.sum(a_out * P * H, dim=1, keepdim=True)
    I = n0_tensor.unsqueeze(-1) + interference - a_out * P * H
    SINR = (a_out * P * H) / (I + 1e-20)

    # 总速率
    rate = torch.sum(torch.log(1 + SINR), dim=(1, 2))
    objective = torch.sum(rate)/10**6  # 最大化平均速率

    # 损失函数（最小化负速率）
    loss = -objective

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练状态
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, "
              f"Avg Rate: {objective.item():.4f}, "
              f"Etas: {np.mean([eta.item() for eta in model.etas])}")

# 保存训练好的模型
torch.save(model.state_dict(), "resource_allocator.pth")

# 1. 首先确保你有相同的模型定义
model = UnrolledResourceAllocator(
    K_layers=K_layers,
    nUE=nUE,
    nRB=nRB,
    init_eta=0.1
)

# 2. 加载保存的模型参数
model.load_state_dict(torch.load("resource_allocator.pth"))

# 3. 将模型设置为评估模式（如果你只是用来推理而不是继续训练）
model.eval()

# 使用示例（假设你有新的输入数据）
# 生成新的问题实例（或使用你自己的数据）
H_np, P_np, n0_np = generate_batch_problem_instance(init_env, env_args, batch_size=1)

# 转换为PyTorch张量
H = torch.tensor(H_np, dtype=torch.float32)
P = torch.tensor(P_np, dtype=torch.float32)
n0 = torch.tensor(n0_np, dtype=torch.float32)

# 初始化a（均匀分配）
a_init = torch.full((1, nUE, nRB), 0.5)  # batch_size=1

# 前向传播（推理）
with torch.no_grad():  # 禁用梯度计算以节省内存
    a_out = model(a_init, H, P, n0)

    # 计算速率
    interference = torch.sum(a_out * P * H, dim=1, keepdim=True)
    I = n0.view(1, 1, 1) + interference - a_out * P * H
    SINR = (a_out * P * H) / (I + 1e-20)
    rate = torch.sum(torch.log(1 + SINR), dim=(1, 2))

    # print(f"Allocated resources: {a_out.squeeze().numpy()}")
    print(f"Achieved rate: {rate.item():.4f} Mbps")