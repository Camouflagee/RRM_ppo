import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic

with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args

NonSequencedEnv = load_env('saved_env/BS1UE20/env.zip')
init_env = SequenceDecisionEnvironmentSB3(env_args)
init_env.__setstate__(NonSequencedEnv.__getstate__())

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
N_rb = env.sce.rbg_Nb  # 每个用户在所有资源块上分配量之和上限
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
H_norm_sq=H
# -------------------------------
# 初始化 a 矩阵满足 a_{k,u}\in[0,1] 且对任一用户 u，有 sum_k a_{k,u} <= N_rb
a = np.random.uniform(0, 1, size=(K, U))
for u in range(U):
    s = a[:, u].sum()
    if s > N_rb:
        a[:, u] = a[:, u] * (N_rb / s)

# 初始化辅助变量 w (与 a 形状一致)
w = np.zeros((K, U))

# 迭代更新参数
max_iters = 500  # 最大迭代次数
step_size_a = 0.05  # 用于 a 更新的步长（梯度上升）
eps = 1e-10  # 防止除 0

# 用于记录目标函数值变化（在二次变换下的目标）
obj_history = []


# -------------------------------
# 定义投影函数: 将单个用户的 K 维向量投影到集合 { z in [0,1]^K, sum(z) <= N_rb }
def project_per_user(x, N_rb):
    """
    将向量 x 分量先剪切到 [0,1]，若总和超过 N_rb，则利用二分查找法求解拉格朗日乘子λ，
    得到投影结果：z_k = clip(x_k - λ, 0, 1) 且 sum(z)=N_rb.
    """
    x = np.clip(x, 0, 1)
    if x.sum() <= N_rb:
        return x
    # 二分法求解 λ，使得 sum(clip(x - λ, 0, 1)) = N_rb
    lam_low, lam_high = 0, np.max(x)
    for _ in range(50):
        lam = (lam_low + lam_high) / 2.0
        z = np.clip(x - lam, 0, 1)
        s = z.sum()
        if s > N_rb:
            lam_low = lam
        else:
            lam_high = lam
    return np.clip(x - (lam_low + lam_high) / 2, 0, 1)


# -------------------------------
# 主循环：交替更新 w 和 a
# -------------------------------
for it in range(max_iters):

    # === Step1. 更新辅助变量 w ===
    # 对于每个资源块 k 和用户 u，
    # 定义干扰项： I_{k,u} = n0 + P[k,u] * sum_{q != u} a[k,q]*H_norm_sq[k,q]
    for k in range(K):
        for u in range(U):
            interference = n0
            for q in range(U):
                if q != u:
                    interference += a[k, q] * P[k, u] * H_norm_sq[k, q]
            # 若 a[k,u] 非常小，w 定义为 0，避免除 0
            if a[k, u] < eps:
                w[k, u] = 0.0
            else:
                w[k, u] = np.sqrt(a[k, u] * P[k, u] * H_norm_sq[k, u]) / (interference + eps)

    # === Step2. 更新变量 a ===
    # 我们采用梯度上升的方法优化二次变换后的目标函数：
    # 对每个 (k,u)，定义：
    #   Q_{k,u} = sqrt(a[k,u] * P[k,u] * H_norm_sq[k,u])
    #   s_{k,u} = 2 * w[k,u] * Q_{k,u} - w[k,u]^2 * I_{k,u}
    # 目标函数： f = sum_{k,u} log(1 + s_{k,u})
    #
    # 注意：a[k,u] 在对应项 f_{k,u} 出现“直接项”以及在其他用户项中作为干扰出现。
    grad_a = np.zeros((K, U))
    current_obj = 0.0

    for k in range(K):
        # 先计算各组干扰项，方便后续使用
        I_vec = np.zeros(U)
        s_vec = np.zeros(U)
        Q_vec = np.zeros(U)
        for u in range(U):
            # 对 (k,u) 计算对应干扰项（注意分母采用 P[k,u]，与原公式一致）
            interference = n0
            for q in range(U):
                if q != u:
                    interference += a[k, q] * P[k, u] * H_norm_sq[k, q]
            I_vec[u] = interference
            # 计算 Q_{k,u} = sqrt(a[k,u]*P[k,u]*H_norm_sq[k,u])
            Q_vec[u] = np.sqrt(a[k, u] * P[k, u] * H_norm_sq[k, u] + eps)
            s_vec[u] = 2 * w[k, u] * Q_vec[u] - (w[k, u] ** 2) * interference
            # 累加目标函数（log(1+s)）
            current_obj += np.log(1 + s_vec[u])

        # 对同一资源块 k，各用户间耦合在梯度中体现
        for u in range(U):
            # --- “直接”梯度：
            # 对 f_{k,u} 关于 a[k,u] 的直接项：
            # d s_{k,u} / d a[k,u] = w[k,u] * sqrt(P[k,u]*H_norm_sq[k,u])/(sqrt(a[k,u]+eps))
            # 梯度 direct = (1/(1+s_{k,u})) * d s_{k,u} / d a[k,u]
            if a[k, u] > eps:
                grad_direct = (w[k, u] * np.sqrt(P[k, u] * H_norm_sq[k, u])) / (np.sqrt(a[k, u] + eps) * (1 + s_vec[u]))
            else:
                grad_direct = 0.0

            # --- “干扰”梯度：
            # 当 a[k,u] 作为干扰项出现在其他用户 v (\(v \neq u\)) 的 f_{k,v} 中：
            # 对于每个 v ≠ u， d s_{k,v} / d a[k,u] = -w[k,v]^2 * P[k,v]*H_norm_sq[k,u]
            # 梯度 干扰项为： - w[k,v]^2*P[k,v]*H_norm_sq[k,u] / (1+s_{k,v])
            grad_interf = 0.0
            for v in range(U):
                if v != u:
                    grad_interf += - (w[k, v] ** 2) * P[k, v] * H_norm_sq[k, u] / (1 + s_vec[v])

            grad_a[k, u] = grad_direct + grad_interf

    obj_history.append(current_obj)

    # 梯度上升更新 a
    a_new = a + step_size_a * grad_a

    # --- 对每个用户 u 的向量 a[:, u] 投影到集合 { z in [0,1]^K, sum(z) <= N_rb }
    for u in range(U):
        a_new[:, u] = project_per_user(a_new[:, u], N_rb)
    # 同时确保每个分量在 [0,1] 内
    a_new = np.clip(a_new, 0, 1)

    a = a_new.copy()

    if it % 10 == 0:
        print(f"Iteration {it}, objective = {current_obj:.4f}")

print("\nFinal objective:", obj_history[-1]*BW//10**6)
print("Final resource allocation a (每行对应一个资源块，各列为不同用户):")
a_opt = a
print(np.array2string(a_opt, separator=', '))
print('done')


# a_opt=[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.],
#  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.],
#  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
#   0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
#   0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 1.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#   0., 0.],
#  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.],
#  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#   0., 0.]]