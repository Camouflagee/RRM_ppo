import copy
import sys

import numpy as np
import yaml
from pydantic import conint

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic, Logger


def quadratic_projection(x_u, N_rb):
    """
    对向量 x_u 进行投影，使得满足以下约束：
    1. 每个元素在 [0, 1] 范围内；
    2. 向量元素总和不超过 N_rb。

    Args:
        x_u (numpy.ndarray): 输入向量，长度为 K。
        N_rb (float): 总和约束的上限。

    Returns:
        numpy.ndarray: 投影后的向量。
    """
    # Step 1: 逐元素剪切到 [0, 1]
    x_u_clipped = np.clip(x_u, 0, 1)

    # Step 2: 检查总和是否小于等于 N_rb
    if np.sum(x_u_clipped) <= N_rb:
        return x_u_clipped

    # Step 3: 投影到集合 {z ∈ [0, 1]^K : sum(z) <= N_rb}
    # 使用二分法寻找合适的 lambda
    def calculate_sum(lambda_val):
        """计算给定 lambda 时的投影和."""
        return np.sum(np.clip(x_u_clipped - lambda_val, 0, 1))

    # 二分法的上下限
    lambda_low, lambda_high = 0, max(x_u_clipped)  # 初始范围
    epsilon = 1e-6  # 精度

    # 开始二分查找
    while lambda_high - lambda_low > epsilon:
        lambda_mid = (lambda_low + lambda_high) / 2
        current_sum = calculate_sum(lambda_mid)
        if current_sum > N_rb:
            lambda_low = lambda_mid  # 说明 lambda 需要更大
        else:
            lambda_high = lambda_mid  # 说明 lambda 需要更小

    # 最终的 lambda 值
    lambda_opt = (lambda_low + lambda_high) / 2
    # 根据最终的 lambda 投影
    z_u = np.clip(x_u_clipped - lambda_opt, 0, 1)

    return z_u


def randomized_round_projection(x, N_rb):
    """
    将向量x在[0,1]内（先clip），然后通过随机化舍入将其投影到{0,1}^K，
    且约束sum(z)<=N_rb。算法：
      1. Clip x到[0,1];
      2. 定义 p = x / sum(x) 为概率分布;
      3. 从 {0,1,...,K-1} 中采样size=N_rb（不放回），采样概率依 p;
      4. 返回对应位置置1，其它置0的二值向量。
    """
    x = np.clip(x, 0, 1)
    # 如果所有x都为0或总和过小，则直接返回全0向量
    if np.sum(x) < 1e-6:
        return np.zeros_like(x)

    # 归一化得到概率分布
    p = x / x.sum()

    # 如果候选数目不足N_rb，则可以直接选择所有非零值所在位置
    nonzero_idx = np.where(x > 1e-6)[0]
    if nonzero_idx.size <= N_rb:
        z = np.zeros_like(x)
        z[nonzero_idx] = 1
        return z

    # 从0到K-1中采样N_rb个索引，不放回
    chosen = np.random.choice(len(x), size=N_rb, replace=False, p=p)
    z = np.zeros_like(x)
    z[chosen] = 1
    return z


def gumbel_softmax_round_projection(x, N_rb, tau=0.5):
    """
    利用Gumbel-Softmax方法实现离散化近似：
    x: 输入向量（未做softmax，作为logits）。这里假设 x>0。
    N_rb: 需要选取1的个数
    tau: 温度参数，tau 越小离散化效果越好。

    返回：
      一个离散的向量 z in {0,1}^K，其中选取的索引为前N_rb个最大值（前向离散）。
    """
    # 加入Gumbel噪声: g = -log(-log(U)), U ~ Uniform(0,1)
    U = np.random.rand(*x.shape)
    g = -np.log(-np.log(U + 1e-10) + 1e-10)

    # 计算带噪声的logits
    noisy_logits = (np.log(x + 1e-10) + g) / tau
    # 计算softmax
    y = np.exp(noisy_logits) / np.sum(np.exp(noisy_logits))

    # 前向离散：选取top N_rb个
    # 这里直接按 y 的值排序取最高的N_rb个
    indices = np.argsort(-y)[:N_rb]
    z = np.zeros_like(y)
    z[indices] = 1
    return z


# 定义投影函数，对每个用户独立投影到 [0,1]^K 且 sum(a[:,u]) <= N_rb
def continue_projection(a_u, N_rb):
    # a_u: 分配给单个用户的 K 维向量
    # 首先将值限制在[0,1]
    a_u = np.clip(a_u, 0, 1)
    s = a_u.sum()
    if s <= N_rb:
        return a_u
    # 若超出上限，则做归一化投影
    return a_u * (N_rb / s)


def discrete_project_per_user(x, N_rb):
    """
    将向量 x 在欧几里得距离意义下投影到二值集合 {0,1}^K，
    同时满足 sum(x) <= N_rb.

    求解问题:
       min_{z in {0,1}^K} ||z - x||^2
       s.t.  sum(z) <= N_rb.

    过程说明:
      1. 对于每个分量 x_i，当 x_i > 0.5 设为1更有利，否则设为0。
      2. 如果得到的1的个数超过 N_rb，则只取 x_i 值最大的 N_rb 项。
    """
    x = np.clip(x, 0, 1)  # 保证 x 在 [0,1] 内（可选步骤）

    # 按 0.5 的阈值判断
    candidate = np.where(x > 0.5)[0]  # 满足 x_i > 0.5 的索引
    z = np.zeros_like(x)

    if candidate.size <= N_rb:
        z[candidate] = 1
    else:
        # 当候选数目超过 N_rb，选择 x_i 较大的 N_rb 个
        # 排序 candidate 中的 x 值，从大到小排序
        sorted_idx = candidate[np.argsort(-x[candidate])]
        chosen_idx = sorted_idx[:N_rb]
        z[chosen_idx] = 1
    return z


with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args
#
# NonSequencedEnv = load_env('saved_env/BS1UE20/env.zip')
# init_env = SequenceDecisionEnvironmentSB3(env_args)
# init_env.__setstate__(NonSequencedEnv.__getstate__())
# env_path_list = [
#     'Experiment_result/seqPPOcons/UE5RB10/ENV/env.zip'
# ]
burst_prob = 0.8
is_H_noise = True
isBurst = False
for idx, (nUE, nRB) in enumerate(
        zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    if idx in [0, 1, 2, ]:
        continue
    np.random.seed(0)
    res = []
    res_proj = []
    num_pair = []
    adaptive_h_error_obj_list = []
    print()
    print("*" * 30, f"场景: UE{nUE}RB{nRB}", "*" * 30)
    # logger = Logger(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/baseline_output.txt')
    env_path = f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip'
    init_env = load_env(env_path)

    test_num = 5
    _error_percent_list = [0, 0.05, 0.1, 0.2] if is_H_noise else [0]
    # _error_percent_list = [0]
    for _error_percent in _error_percent_list:
        print("=" * 10, f"场景: UE{nUE}RB{nRB} - error_percent: {_error_percent}", "=" * 10)
        print(f'env_path: {env_path}')
        error_percent = _error_percent
        env = init_env
        for loop in range(test_num):
            # logger = Logger(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/baseline_output.txt')
            # sys.stdout = logger
            # ============================
            # 1. 参数设置
            # ============================
            # 设定资源块数目 K, 用户数 U
            K = env.nRB  # 例如：3个资源块
            U = env.nUE  # 例如：4个用户
            BW = env.sce.BW
            # 噪声功率 n0 和每用户的资源约束 N_rb
            n0 = env.get_n0()  # 噪声功率
            N_rb = 20
            # env.sce.rbg_Nb) if env.sce.rbg_Nb is not None else env.sce.Nrb  # 每个用户在所有资源块上分配量之和上限
            obs, info = env.reset_onlyforbaseline()

            # 因为集合 A（基站索引）只有一个元素，所以我们只考虑该基站
            # 生成示例参数：功率 P_{b,k,u} 和信道增益 ||H_{b,k,u}||^2（这里直接用正数表示）
            # seed = np.random.randint(low=0, high=99)
            # np.random.seed(0)  # 固定随机种子，保证结果可重复
            P_constant = env.BSs[0].Transmit_Power()
            P = np.ones((K, U)) * P_constant
            H_dB = info['CSI']
            H_uk = 10 ** (H_dB / 10)  # info['CSI']: unit dBm
            H = (1 / H_uk).reshape(U, K).transpose()

            if isBurst and burst_prob:
                user_burst = np.random.rand(nUE) < burst_prob  # Shape: (nUE,)
                user_burst_mat = np.repeat(user_burst[None, :], nRB, axis=0)
            # user_burst=np.random.rand(nUE) < burst_prob
            if is_H_noise:
                H_error_dB = env.get_estimated_H(H_dB, _error_percent)  # add 5% estimated error
                H_error_uk = 10 ** (H_error_dB / 10)
                H_error = (1 / H_error_uk).reshape(U, K).transpose()
                H_norm_sq = H_error  # This H is used by algorithm
            else:
                H_norm_sq = H  # This H is used by algorithm
            # todo check
            import numpy as np

            # -------------------------------
            # 参数设置（示例设定）
            # -------------------------------

            #
            # 信道 H_{k,u}（这里假设为标量，实际中可能为向量或矩阵，此处用其模平方）
            # H_norm_sq = H

            # 梯度上升参数
            max_iters = 300
            eta = 0.01  # 学习率
            tol = 1e-4  # 收敛阈值

            # -------------------------------
            # 初始化 a_{k,u} 满足约束: a在[0,1]内，且每个用户对所有资源块的分配和不超过 N_rb
            # 初始化时可以均匀分配或随机初始化
            a = np.random.uniform(0, 1, size=(K, U))
            for u in range(U):
                if a[:, u].sum() > N_rb:
                    a[:, u] = a[:, u] * (N_rb / a[:, u].sum())


            def projection(a_u, N_rb):
                # the projection used in the algorithm
                return continue_projection(a_u, N_rb)


            # 计算目标函数值
            def compute_rate(a, P, _H, n0, _user_burst_mat=None):
                rate = 0
                K, U = a.shape
                if _user_burst_mat:
                    _H = _H * _user_burst_mat
                for k in range(K):
                    for u in range(U):
                        # if isBurst and user_burst[U] == 0:  # 如果用户没有数据请求，跳过
                        #     continue
                        # 计算干扰项I_{k,u}
                        inter = 0
                        for up in range(U):
                            if up != u:
                                inter += a[k, up] * P[k, up] * _H[k, up]
                        I_ku = inter + n0
                        gamma = (a[k, u] * P[k, u] * _H[k, u]) / I_ku
                        rate += np.log(1 + gamma)
                return rate


            # 计算目标函数相对于 a 的梯度
            def compute_grad(a, P, _H, n0):
                K, U = a.shape
                grad = np.zeros((K, U))
                # 预先计算每个 (k,u) 的 I_{k,u} 和 gamma_{k,u}
                I = np.zeros((K, U))
                gamma = np.zeros((K, U))

                for k in range(K):
                    for u in range(U):
                        inter = 0
                        for up in range(U):
                            if up != u:
                                inter += a[k, up] * P[k, up] * _H[k, up]
                        I[k, u] = inter + n0
                        gamma[k, u] = (a[k, u] * P[k, u] * _H[k, u]) / I[k, u]

                # 对于任意 (k,u) 计算梯度：
                for k in range(K):
                    for u in range(U):
                        # 部分1：对其本身项
                        d_gamma_d_aku = P[k, u] * H_norm_sq[k, u] / I[k, u]
                        grad[k, u] += (1 / (1 + gamma[k, u])) * d_gamma_d_aku

                        # 部分2：当 a[k,u] 出现在其他用户同一资源块 k 干扰项中
                        for up in range(U):
                            if up == u:
                                continue
                            # 对于 (k, up)，分母 I[k,up] 对 a[k,u] 的影响
                            # I[k,up] = \sum_{v \neq up} a[k, v] * P[k, v]*H_norm_sq[k, v] + n0，
                            # 对 a[k,u] 的偏导为 P[k, u]*H_norm_sq[k, u]
                            dI_d_aku = P[k, u] * H_norm_sq[k, u]
                            # gamma[k,up] = (a[k, up]*P[k, up]*H_norm_sq[k, up]) / I[k,up],
                            # 关于 a[k,u] 的偏导（负向）：
                            d_gamma_d_aku = - (a[k, up] * P[k, up] * H_norm_sq[k, up] * dI_d_aku) / (I[k, up] ** 2)
                            grad[k, u] += (1 / (1 + gamma[k, up])) * d_gamma_d_aku
                return grad


            # -------------------------------
            # 主循环：投影梯度上升
            # -------------------------------
            rate_history = []
            for iter in range(max_iters):
                current_rate = compute_rate(a, P, H_norm_sq, n0)
                rate_history.append(current_rate * BW // 10 ** 6)

                grad = compute_grad(a, P, H_norm_sq, n0)
                a_new = a + eta * grad

                # 对每个用户分别投影到可行域
                for u in range(U):
                    a_new[:, u] = projection(a_new[:, u], N_rb)

                # 终止条件：若目标函数增量很小则退出
                if np.abs(compute_rate(a_new, P, H_norm_sq, n0) - current_rate) < tol:
                    a = a_new
                    # print(f"Converged at iter {iter}")
                    break
                a = a_new

            a_opt = a
            a_opt_discrete = copy.deepcopy(a)
            for u in range(U):
                a_opt_discrete[:, u] = discrete_project_per_user(a_opt_discrete[:, u],
                                                                 N_rb)  # randomized_round_project
            opt_obj = compute_rate(a_opt, P, H, n0) * BW // 10 ** 6
            res.append(opt_obj)
            opt_obj_discrete = compute_rate(a_opt_discrete, P, H, n0) * BW // 10 ** 6
            res_proj.append(opt_obj_discrete)
            num_pair.append(sum(sum(a_opt_discrete)))
            if is_H_noise:
                adaptive_h_error_obj = compute_rate(a_opt, P, H_norm_sq, n0) * BW // 10 ** 6
                adaptive_h_error_obj_list.append(adaptive_h_error_obj)
            # print("最优目标值：", opt_obj)
            # print("最优资源分配 a_opt:")
            # print(np.array2string(a_opt, separator=', '))
            # print("投影到离散0-1可行域后最终目标值：", opt_obj_discrete)
            # print("配对长度: ", sum(sum(a_opt_discrete)))
            # print("投影到离散可行域 a_opt_discrete:")
            # print(np.array2string(a_opt_discrete, separator=', '))
            # print("="*20,"done:","="*20)
        print(f"{test_num}次实验平均后结果")
        if is_H_noise:
            print("噪声估计信道上最终目标值：", np.mean(adaptive_h_error_obj_list))
        print(f"真实信道上最终目标值：{np.mean(res):.2f}", )
        print(f"真实信道上投影后最终目标值：{np.mean(res_proj):.2f}", )
        print(f"UE/RB配对数量: {np.mean(num_pair):.2f}", )
        # print("=" * 20, "done:", "=" * 20)
