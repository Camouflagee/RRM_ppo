from torch import nn
import torch


class ResourceAllocationUnrolled(nn.Module):
    def __init__(self, num_layers, K, U):
        super().__init__()
        self.num_layers = num_layers
        self.K = K  # 资源块数
        self.U = U  # 用户数

        # 可学习参数：每层的步长
        self.step_sizes = nn.Parameter(torch.ones(num_layers) * 0.05)

        # 可学习参数：初始化矩阵
        self.a_init = nn.Parameter(torch.rand(K, U))

        # 可学习阈值参数（用于投影）
        self.threshold_params = nn.Parameter(torch.ones(num_layers, U))


    def forward_layer(self, a_prev, w, H, P, n0, layer_idx):
        # 计算干扰项 I_{k,u}
        interference = n0 + torch.sum(a_prev * P * H, dim=1, keepdim=True) - a_prev * P * H

        # 计算梯度 (文档2中的grad_a)
        Q = torch.sqrt(a_prev * P * H + 1e-10)
        s = 2 * w * Q - w ** 2 * interference

        # 直接梯度项
        grad_direct = w * torch.sqrt(P * H) / (torch.sqrt(a_prev + 1e-10) * (1 + s))

        # 干扰梯度项
        grad_interf = -torch.sum(
            (w ** 2 * P.unsqueeze(2) * H.unsqueeze(1)) / (1 + s.unsqueeze(2)),
            dim=1
        ) + (w ** 2 * P * H) / (1 + s)

        # 合并梯度
        grad = grad_direct + grad_interf

        # 应用可学习步长
        a_tilde = a_prev + self.step_sizes[layer_idx] * grad

        # 投影到可行域
        a_next = self.project(a_tilde, layer_idx)

        return a_next


    def project(self, a_tensor, layer_idx):
        # 确保在[0,1]范围内
        a_clipped = torch.clamp(a_tensor, 0, 1)

        # 获取当前层的阈值参数
        thresholds = torch.sigmoid(self.threshold_params[layer_idx])

        # 对每个用户进行投影
        for u in range(self.U):
            user_vec = a_clipped[:, u]

            # 计算当前分配总和
            current_sum = torch.sum(user_vec)

            if current_sum > self.N_rb:
                # 使用可学习阈值进行软投影
                sorted_vals, _ = torch.sort(user_vec, descending=True)
                cumsum = torch.cumsum(sorted_vals, dim=0)
                k = torch.argmax((cumsum - self.N_rb) / (torch.arange(1, self.K + 1)) <= thresholds[u])

                # 应用投影
                lambda_val = (cumsum[k] - self.N_rb) / (k + 1)
                user_vec = torch.clamp(user_vec - lambda_val, 0, 1)

        return a_clipped
    def compute_w(self, a, H, P, n0):
        interference = n0 + torch.sum(a * P * H, dim=1, keepdim=True) - a * P * H
        w = torch.sqrt(a * P * H + 1e-10) / (interference + 1e-10)
        return w


    def forward(self, H, P, n0):
        # 初始化a
        a = torch.sigmoid(self.a_init)  # 映射到[0,1]

        # 迭代过程
        for i in range(self.num_layers):
            w = self.compute_w(a, H, P, n0)
            a = self.forward_layer(a, w, H, P, n0, i)

        return a


def loss_fn(a_pred, H_true, P, n0):
    # 计算实际SINR
    signal = a_pred * P * H_true
    interference = torch.sum(a_pred * P * H_true, dim=1, keepdim=True) - signal + n0
    sinr = signal / (interference + 1e-10)

    # 计算和速率
    sum_rate = torch.sum(torch.log2(1 + sinr))

    return -sum_rate  # 最大化目标


model = ResourceAllocationUnrolled(num_layers=10, K=30, U=15)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import numpy as np
from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic
import yaml


def generate_channel_state():
    """
    生成无线通信系统的信道状态信息

    返回:
        H: 信道增益矩阵 (K x U)
        P: 发射功率矩阵 (K x U)
        n0: 噪声功率 (标量)
        N_rb: 每个用户的最大资源块数 (标量)
    """
    # 加载环境配置
    with open('config/config_environment_setting.yaml', 'r') as file:
        env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))

    sce = env_args
    # 选择特定场景 (UE=12, RB=30)
    nUE, nRB = 12, 30

    # 加载预训练环境
    init_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')
    env = init_env

    # 获取系统参数
    K = env.nRB  # 资源块数
    U = env.nUE  # 用户数
    N_rb = nRB // 2  # 每个用户的最大资源块数

    # 重置环境获取初始状态
    obs, info = env.reset_onlyforbaseline()

    # 获取发射功率 (所有用户和资源块使用相同功率)
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((K, U)) * P_constant

    # 获取信道状态信息 (CSI) 并转换为线性值
    H_uk = 10 ** (info['CSI'] / 10)  # 从dBm转换为线性值

    # 重塑信道矩阵 (U x K) 并转置为 (K x U)
    H = (1 / H_uk).reshape(U, K).transpose()

    # 获取噪声功率
    n0 = env.get_n0()

    return H, P, n0, N_rb
for epoch in range(1000):
    H, P, n0, _ = generate_channel_state()  # 生成信道状态
    P = torch.from_numpy(P).float()
    # n0 = torch.from_numpy(n0).float()
    H = torch.from_numpy(H).float()
    a_pred = model(H, P, n0)
    loss = loss_fn(a_pred, H, P, n0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()