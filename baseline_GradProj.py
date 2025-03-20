import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic, Logger

with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args


def discrete_project_per_user(x, N_rb):
    """
    将向量 x 投影到 {0,1}^K 且 sum(x)<=N_rb 的集合中
    """
    x = np.clip(x, 0, 1)
    candidate = np.where(x > 0.5)[0]
    z = np.zeros_like(x)
    if candidate.size <= N_rb:
        z[candidate] = 1
    else:
        sorted_idx = candidate[np.argsort(-x[candidate])]
        chosen_idx = sorted_idx[:N_rb]
        z[chosen_idx] = 1
    return z


#
# NonSequencedEnv = load_env('saved_env/BS1UE20/env.zip')
# init_env = SequenceDecisionEnvironmentSB3(env_args)
# init_env.__setstate__(NonSequencedEnv.__getstate__())
# env_path_list = [
#     'Experiment_result/seqPPOcons/UE5RB10/ENV/env.zip'
# ]
sol_dict={}
for idx, (nUE, nRB) in enumerate(
        zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    logger = Logger(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/baseline_output.txt')
    init_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')
    # ============================
    # 1. 参数设置
    # ============================

    env = init_env
    # 设定资源块数目 K, 用户数 U
    K = env.nRB  # 例如：3个资源块
    U = env.nUE  # 例如：4个用户
    BW = env.sce.BW
    # 噪声功率 n0 和每用户的资源约束 N_rb
    n0 = env.get_n0()  # 噪声功率
    N_rb = env.sce.rbg_Nb if env.sce.rbg_Nb is not None else env.sce.Nrb  # 每个用户在所有资源块上分配量之和上限
    obs, info = env.reset_onlyforbaseline()

    # 因为集合 A（基站索引）只有一个元素，所以我们只考虑该基站
    # 生成示例参数：功率 P_{b,k,u} 和信道增益 ||H_{b,k,u}||^2（这里直接用正数表示）
    # seed = np.random.randint(low=0, high=99)
    np.random.seed(0)  # 固定随机种子，保证结果可重复
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((K, U)) * P_constant
    H_uk = 10 ** (info['CSI'] / 10)  # info['CSI']: unit dBm
    H = (1 / H_uk).reshape(U, K).transpose()

    import numpy as np

    # -------------------------------
    # 参数设置（示例设定）
    # -------------------------------
    np.random.seed(0)

    # 信道 H_{k,u}（这里假设为标量，实际中可能为向量或矩阵，此处用其模平方）
    H_norm_sq = H

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


    # 定义投影函数，对每个用户独立投影到 [0,1]^K 且 sum(a[:,u]) <= N_rb
    def projection(a_u, N_rb):
        # a_u: 分配给单个用户的 K 维向量
        # 首先将值限制在[0,1]
        a_u = np.clip(a_u, 0, 1)
        s = a_u.sum()
        if s <= N_rb:
            return a_u
        # 若超出上限，则做归一化投影
        return a_u * (N_rb / s)


    # 计算目标函数值
    def compute_rate(a, P, H_norm_sq, n0):
        rate = 0
        K, U = a.shape
        for k in range(K):
            for u in range(U):
                # 计算干扰项I_{k,u}
                inter = 0
                for up in range(U):
                    if up != u:
                        inter += a[k, up] * P[k, up] * H_norm_sq[k, up]
                I_ku = inter + n0
                gamma = (a[k, u] * P[k, u] * H_norm_sq[k, u]) / I_ku
                rate += np.log(1 + gamma)
        return rate


    # 计算目标函数相对于 a 的梯度
    def compute_grad(a, P, H_norm_sq, n0):
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
                        inter += a[k, up] * P[k, up] * H_norm_sq[k, up]
                I[k, u] = inter + n0
                gamma[k, u] = (a[k, u] * P[k, u] * H_norm_sq[k, u]) / I[k, u]

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
        rate_history.append(current_rate)

        grad = compute_grad(a, P, H_norm_sq, n0)
        a_new = a + eta * grad

        # 对每个用户分别投影到可行域
        for u in range(U):
            a_new[:, u] = projection(a_new[:, u], N_rb)

        # 终止条件：若目标函数增量很小则退出
        if np.abs(compute_rate(a_new, P, H_norm_sq, n0) - current_rate) < tol:
            a = a_new
            print(f"Converged at iter {iter}")
            break
        a = a_new

    # print("最终目标值：", compute_rate(a, P, H_norm_sq, n0) * BW // 10 ** 6)
    # print("最终资源分配 a_opt:")
    # a_opt = a
    # print(np.array2string(a_opt, separator=', '))
    print("=" * 10, f"UE{nUE}RB{nRB}场景", "=" * 10)
    a_opt_discret = a
    for u in range(U):
        a_opt_discret[:, u] = discrete_project_per_user(a_opt_discret[:, u], N_rb)
    print("最终目标值：", env.cal_sumrate_givenH(a_opt_discret.reshape(K, U).transpose(), info['CSI'])[0])
    print("a_opt_discret:")
    print(np.array2string(a_opt_discret, separator=', '))
    sol_dict.update({
        f'UE{nUE}RB{nRB}':
            {
                'sol': a,
                'H': info['CSI'],
                'K': K,
                'U': U,
            }
    }
    )
print('done')

