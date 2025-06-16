import numpy as np
import yaml
from click.core import batch
from torch import nn
import torch

from utils import DotDic, load_env

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ResourceAllocationDataset(Dataset):
    def __init__(self, init_env, env_args, dataset_size=10000, random_walk=False):
        """
        初始化资源分配数据集

        参数:
            init_env: 初始化环境对象
            env_args: 环境参数
            dataset_size: 数据集大小（样本数量）
            random_walk: 是否使用随机行走生成数据
        """
        self.dataset_size = dataset_size
        self.random_walk = random_walk
        self.init_env = init_env
        self.env_args = env_args

        # 预生成所有数据（可改为惰性加载）
        self.H, self.P, self.n0, self.a_opt = self._generate_all_data()

    def _generate_all_data(self):
        """一次性生成所有数据样本"""
        # 分批生成数据以避免内存溢出
        batch_size = min(100, self.dataset_size)
        num_batches = int(np.ceil(self.dataset_size / batch_size))

        all_H, all_P, all_n0, all_a_opt = [], [], [], []

        for _ in range(num_batches):
            actual_batch_size = min(batch_size, self.dataset_size - len(all_H))
            H_batch, P_batch, n0_batch, a_opt_batch = generate_batch_problem_instance(
                self.init_env, self.env_args,
                batch_size=actual_batch_size,
                random_walk=self.random_walk
            )
            all_H.append(H_batch)
            all_P.append(P_batch)
            all_n0.append(n0_batch)
            all_a_opt.append(a_opt_batch)

        # 沿批次维度拼接
        H = np.concatenate(all_H, axis=0)
        P = np.concatenate(all_P, axis=0)
        n0 = np.concatenate(all_n0, axis=0)
        a_opt = np.concatenate(all_a_opt, axis=0)

        return H, P, n0, a_opt

    def __len__(self):
        """返回数据集大小"""
        return self.dataset_size

    def __getitem__(self, idx):
        """返回单个样本"""
        return {
            'H': self.H[idx].astype(np.float32),
            'P': self.P[idx].astype(np.float32),
            'n0': np.array([self.n0[idx]]).astype(np.float32),  # 确保n0为标量
            'a_init': np.full((nUE, nRB), 0.5).astype(np.float32),  # 固定初始分配
            'a_opt': self.a_opt[idx].astype(np.float32)
        }
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
    H_uk = 10 ** (info['CSI'] / 10)  # 从dBm转换为线性值
    H = H_uk.reshape(U, K)
    # 获取噪声功率
    n0 = env.get_n0()

    a_init = np.random.rand(K, U)  # 随机(0,1)
    a = a_init
    # 参数设置
    Nk = K  # 资源块数
    Nu = U  # 用户数
    max_iter = 100  # 最大迭代次数
    tol = 1e-4  # 收敛容忍度
    H_sq = (1/H).transpose(1,0)
    P_cal = P.transpose(1,0)
    for iter in range(max_iter):
        # ==============================================================================
        # 计算当前 gamma 和 A
        # gamma = np.zeros((Nk, Nu))
        # A = np.zeros((Nk, Nu))
        # for k in range(Nk):
        #     for u in range(Nu):
        #         c_ku = P[k, u] * H_sq[k, u]
        #         # 计算干扰项
        #         interference = 0
        #         for uprime in range(Nu):
        #             if uprime != u:
        #                 d_ku_prime = P[k, uprime] * H_sq[k, uprime]
        #                 interference += a[k, uprime] * d_ku_prime
        #         denominator = interference + n0
        #         gamma_ku = (a[k, u] * c_ku) / denominator if denominator != 0 else 0
        #         gamma[k, u] = gamma_ku
        #         A_ku = a[k, u] * c_ku + interference + n0
        #         A[k, u] = A_ku
        # 计算系数
        # coeff = np.zeros((Nk, Nu))
        # for k in range(Nk):
        #     for u in range(Nu):
        #         c_ku = P[k, u] * H_sq[k, u]
        #         term1 = c_ku / A[k, u] if A[k, u] != 0 else 0
        #         term2 = 0
        #         for uprime in range(Nu):
        #             if uprime != u:
        #                 gamma_ku_prime = gamma[k, uprime]
        #                 d_ku_prime = P[k, uprime] * H_sq[k, uprime]
        #                 A_ku_prime = A[k, uprime]
        #                 term2 += (gamma_ku_prime * d_ku_prime) / A_ku_prime if A_ku_prime != 0 else 0
        #         coeff[k, u] = term1 - term2
        # ==============================================================================

        # =============向量化加速=============
        C = P_cal * H_sq  # c_ku for all k, u
        # 计算干扰项 (对于每个k,u，计算sum_{u'≠u} a[k,u']*P[k,u']*H_sq[k,u'])
        # 使用广播技巧
        interference = (a * C).sum(axis=1, keepdims=True) - a * C

        # 计算gamma
        denominator = interference + n0
        gamma = np.where(denominator != 0, (a * C) / denominator, 0)

        # 计算A
        A = a * C + interference + n0

        # 计算系数
        # term1 = C / A (with 0 where A is 0)
        term1 = np.where(A != 0, C / A, 0)

        # term2 = sum_{u'≠u} (gamma[k,u'] * C[k,u']) / A[k,u']
        # 对于每个k,u，计算sum_{u'≠u} gamma[k,u']*C[k,u']/A[k,u']
        # 首先计算每个元素的贡献
        contrib = np.where(A != 0, gamma * C / A, 0)
        # 然后对每个k，计算所有u'≠u的和
        term2 = contrib.sum(axis=1, keepdims=True) - contrib

        coeff = term1 - term2
        # 向量化加速代码结束

        # 更新a，根据系数决定0或1
        a_new = np.where(coeff > 0, 1, 0)

        # 检查收敛
        if np.max(np.abs(a_new - a)) < tol:
            # print(f"收敛于第 {iter} 次迭代")
            break
        a = a_new.copy()

    return H, P, n0, a



import torch
import torch.nn as nn

import numpy as np
from utils import DotDic  # 假设DotDic在你的环境中可用


class UnrolledResourceAllocator(nn.Module):
    def __init__(self, K_layers, nUE, nRB, init_eta=0.05, eps=1e-20):
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
        # a = a_init.clone()
        # batch_size = a.size(0)
        #
        # # 确保n0正确广播 - 修正点
        # if n0.dim() == 0:
        #     # 标量 -> [1, 1, 1]
        #     n0_tensor = n0.view(1, 1, 1).expand(batch_size, 1, 1)
        # elif n0.dim() == 1:
        #     # [batch] -> [batch, 1, 1]
        #     n0_tensor = n0.view(batch_size, 1, 1)
        # else:
        #     # 已经是 [batch, 1, 1] 或兼容形状
        #     n0_tensor = n0

        # 迭代K次（网络层数）
        # for t in range(self.K_layers):
        #     # 步骤1：更新辅助变量w
        #     # 计算干扰+噪声：I = n0 + Σ_{u'≠u} a_{k,u'} * P_{k,u'} * |H_{k,u'}|^2
        #     # 注意: 广播用于批次处理 [batch, UE, RB] -> [batch, 1, RB]
        #     interference = torch.sum(a * P * H, dim=1, keepdim=True)  # 总干扰
        #     I = n0_tensor  + interference - a * P * H  # 排除当前用户的自身干扰
        #
        #     # 计算辅助变量 w = sqrt(a * P * H) / I
        #     w = torch.sqrt(a * P * H + self.eps) / (I + self.eps)
        #
        #
        #
        #     # 步骤2：更新分配矩阵a
        #     # 计算中间变量 s
        #     s = 2 * w * torch.sqrt(a * P * H + self.eps) - w.pow(2) * I
        #
        #     # 计算梯度直接项
        #     g_direct = w * torch.sqrt(P * H) / ((1 + s) * torch.sqrt(a + self.eps))
        #
        #     # 计算梯度干扰项
        #     # 计算辅助矩阵 Q = w^2 * P / (1 + s)
        #     Q = w.pow(2) * P / (1 + s + self.eps)
        #
        #     # 对所有用户求和（同一资源块上）
        #     Q_sum = torch.sum(Q, dim=1, keepdim=True)  # [batch, 1, RB]
        #
        #     # 计算干扰项: g_interf = -H * (Q_sum - Q)
        #     g_interf = -H * (Q_sum - Q)
        #
        #     # 完整梯度 = 直接项 + 干扰项
        #     g = g_direct + g_interf
        #
        #     # 梯度上升更新: a = a + eta * gradient
        #     a = a + self.etas[t] * g
        #
        #     # 可选的: 数值稳定性处理（防止溢出）
        #     a = torch.clamp(a, 0.0, 1.0)
        # 计算信道增益平方 |h|^2
        batch_size = a_init.shape[0]
        H_norm_sq = H.transpose(dim0=1,dim1=2)  # [batchsize, K, U]
        a = a_init.clone().transpose(dim0=1,dim1=2)
        P = P.transpose(dim0=1,dim1=2)
        # 如果 n0 是标量，扩展为与 batchsize 兼容的形状
        if np.isscalar(n0):
            n0_expanded = n0
        else:
            # n0 形状为 [batchsize]，扩展为 [batchsize, 1, 1] 用于广播
            n0_expanded = n0.reshape(batch_size,1,1)
        eps = self.eps
        H_norm_sq = 1 / H_norm_sq
        for it in range(K_layers):
            # === Step1. 更新辅助变量 w ===
            # 计算干扰项: I_{k,u} = n0 + P[k,u] * sum_{q != u} a[k,q] * H_norm_sq[k,q]
            # 向量化计算干扰项
            # 计算 a * H_norm_sq: [batchsize, K, U]
            aH = a * H_norm_sq

            # 计算每个资源块上其他用户的总和: sum_{q != u} a[k,q] * H_norm_sq[k,q]
            # 先计算总和，然后减去当前用户u的贡献
            sum_aH = torch.sum(aH, dim=-1, keepdim=True)  # [batchsize, K, 1]
            other_users_sum = sum_aH - aH  # [batchsize, K, U]

            # 计算干扰项: n0 + P * other_users_sum
            interference = n0_expanded + P * other_users_sum  # [batchsize, K, U]

            # 计算分子: sqrt(a * P * H_norm_sq)
            numerator = torch.sqrt(a * P * H_norm_sq + eps)  # [batchsize, K, U]

            # 计算 w，当 a < eps 时设为 0
            w = torch.zeros_like(a)
            mask = a >= eps
            w[mask] = numerator[mask] / (interference[mask] + eps)

            # === Step2. 更新变量 a ===
            # 重新计算干扰项 (与 Step1 相同)
            interference_step2 = n0_expanded + P * other_users_sum

            # 计算 Q_vec = sqrt(a * P * H_norm_sq)
            Q_vec = torch.sqrt(a * P * H_norm_sq + eps)  # [batchsize, K, U]

            # 计算 s_vec = 2 * w * Q_vec - w^2 * interference_step2
            s_vec = 2 * w * Q_vec - torch.square(w) * interference_step2  # [batchsize, K, U]

            # 计算直接梯度部分
            # grad_direct = (w * sqrt(P * H_norm_sq)) / (sqrt(a) * (1 + s_vec))
            sqrt_PH = torch.sqrt(P * H_norm_sq + eps)  # [batchsize, K, U]
            sqrt_a = torch.sqrt(a + eps)  # [batchsize, K, U]
            grad_direct = (w * sqrt_PH) / (sqrt_a * (1 + s_vec + eps))

            # 当 a < eps 时，直接梯度设为 0
            grad_direct[a < eps] = 0.0

            # 计算干扰梯度部分
            # 计算中间项: A = (w^2 * P) / (1 + s_vec)
            A = (torch.square(w) * P) / (1 + s_vec + eps)  # [batchsize, K, U]

            # 计算每个资源块上其他用户的 A 总和
            sum_A = torch.sum(A, dim=-1, keepdim=True)  # [batchsize, K, 1]
            other_A_sum = sum_A - A  # [batchsize, K, U]

            # 计算干扰梯度: grad_interf = -H_norm_sq * other_A_sum
            grad_interf = -H_norm_sq * other_A_sum  # [batchsize, K, U]

            # 合并梯度
            grad_a = grad_direct + grad_interf  # [batchsize, K, U]
            # assert not torch.any(torch.isnan(grad_a))
            # 更新 a
            a = a + self.etas[it] * grad_a
            a = torch.clamp(a, min=0.0, max=1.0)

        return a


def generate_batch_problem_instance(init_env,env_args, batch_size=32, random_walk=False):
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
    batch_aopt = []
    for _ in range(batch_size):
        if random_walk:
            init_env.random_walk()
        H, P, n0, a_opt = generate_problem_instance(init_env, env_args)  # 调用原始单实例函数
        batch_H.append(H)
        batch_P.append(P)
        batch_n0.append(n0)
        batch_aopt.append(a_opt)
    return np.array(batch_H), np.array(batch_P), np.array(batch_n0), np.array(batch_aopt)


# 训练配置
nUE = 12
nRB = 30
K_layers = 50  # 展开层数
n_epochs = 3000
batch_size = 32
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
train_model=True
if train_model:

    # 训练循环
    H_np, P_np, n0_np, aopt_np = generate_batch_problem_instance(init_env, env_args, batch_size, False)

    for epoch in range(n_epochs):
        # 生成批量问题实例
        # H_np, P_np, n0_np, aopt_np = generate_batch_problem_instance(init_env, env_args, batch_size)

        # 转换为PyTorch张量
        H = torch.tensor(H_np, dtype=torch.float32)
        P = torch.tensor(P_np, dtype=torch.float32)
        n0 = torch.tensor(n0_np, dtype=torch.float32)
        a_opt = torch.tensor(aopt_np, dtype=torch.float32)
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
        # a_out = a_out.transpose(1,2)
        # signal_power=a_out * P / H
        # interference = torch.sum(signal_power, dim=1, keepdim=True)
        # interference_m=torch.tile(interference, dims=(1,12,1))
        # I = n0_tensor + interference_m - signal_power
        # SINR = (signal_power) / (I + 1e-30)
        #
        # # 总速率
        # rate = torch.sum(torch.log2(1 + SINR), dim=(1, 2))
        # objective = (torch.mean(rate)* 180000)/(10**6)  # 最大化平均速率

        # 损失函数（最小化负速率）
        loss = torch.mean(torch.square(a_out - a_opt))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练状态
        if (epoch + 1) % 10 == 0:
            sum_rate = np.mean([init_env.cal_sumrate_givenH(a_out.transpose(1, 2)[i, :, :], 10 * torch.log10(H[i, :, :]))[0] for i in range(batch_size)])
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, "
                  f"sum_rate: {sum_rate:.2f}, "
                  f"Etas: {np.mean([eta.item() for eta in model.etas]):.4f}")
            # print(f"\nFinal objective:{np.mean([init_env.cal_sumrate_givenH(a_out.transpose(1,2)[i,:,:], H[i,:,:])[0] for i in batch_size]):.2f}")

    # 创建数据集
    # dataset = ResourceAllocationDataset(
    #     init_env=init_env,
    #     env_args=env_args,
    #     dataset_size=10000,  # 10,000个样本
    #     random_walk=True
    # )
    # train script version 2
    # # 创建数据加载器
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=32,  # 任意批大小
    #     shuffle=True,  # 随机打乱
    #     num_workers=4,  # 多进程加载
    #     pin_memory=True  # 加快GPU传输
    # )
    #
    # # 训练循环示例
    # for epoch in range(n_epochs):
    #     for batch in dataloader:
    #         # 将数据转移到设备
    #         H = batch['H'].to(device)
    #         P = batch['P'].to(device)
    #         n0 = batch['n0'].to(device)
    #         a_init = batch['a_init'].to(device)
    #         a_opt = batch['a_opt'].to(device)
    #
    #         # 模型前向传播
    #         a_out = model(a_init, H, P, n0.squeeze(1))  # 调整n0形状
    #
    #         # 计算损失
    #         loss = torch.mean(torch.square(a_out - a_opt))
    #
    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # 保存训练好的模型
    torch.save(model.state_dict(), f"resource_allocator_K_layers{K_layers}_nRB{nRB}_nUE{nUE}_4096.pth")

eval_model = True
if eval_model:
    # 1. 首先确保你有相同的模型定义
    model = UnrolledResourceAllocator(
        K_layers=K_layers,
        nUE=nUE,
        nRB=nRB,
        init_eta=0.1
    )

    # 2. 加载保存的模型参数
    model.load_state_dict(torch.load("resource_allocator_K_layers50_nRB30_nUE12.pth"))

    # 3. 将模型设置为评估模式（如果你只是用来推理而不是继续训练）
    model.eval()
    batch_size = 128
    # 使用示例（假设你有新的输入数据）
    # 生成新的问题实例（或使用你自己的数据）
    H_np, P_np, n0_np, aopt_np = generate_batch_problem_instance(init_env, env_args, batch_size=batch_size)

    # 转换为PyTorch张量
    H = torch.tensor(H_np, dtype=torch.float32)
    P = torch.tensor(P_np, dtype=torch.float32)
    n0 = torch.tensor(n0_np, dtype=torch.float32)
    batch_size = n0.shape[0]
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

    # 初始化a（均匀分配）
    a_init = torch.full((batch_size, nUE, nRB), 0.5)  # batch_size=1

    # 前向传播（推理）
    with torch.no_grad():  # 禁用梯度计算以节省内存
        a_out = model(a_init, H, P, n0_tensor)

        sum_rate = np.mean(
            [init_env.cal_sumrate_givenH(a_out.transpose(1, 2)[i, :, :], 10 * torch.log10(H[i, :, :]))[0] for i in
             range(batch_size)])

        print(f"Achieved rate: {sum_rate:.4f} Mbps")
