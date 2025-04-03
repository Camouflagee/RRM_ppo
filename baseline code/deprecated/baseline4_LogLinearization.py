import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic

with open('../../config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args

NonSequencedEnv = load_env('../../saved_env/BS1UE20/env.zip')
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
seed = np.random.randint(low=0,high=999)
np.random.seed(seed)  # 固定随机种子，保证结果可重复
P_constant = env.BSs[0].Transmit_Power()
P = np.ones((K, U)) * P_constant
H_uk = 10 ** (info['CSI'] / 10)  # info['CSI']: unit dBm
H = (1 / H_uk).reshape(U, K).transpose()


def sca_log_rate_maximization(H, P, n0, solver=cp.GUROBI, max_iter=20, tol=1e-3, verbose=False):
    """
    使用 SCA 方法求解:

        max sum_{k,u} log(1 + gamma_{k,u})
        s.t. 0 <= a_{k,u} <= 1

    其中:
      gamma_{k,u} = (a_{k,u} * P[k,u] * |H[k,u]|^2) /
                    ( sum_{u' != u} a_{k,u'} * P[k,u] * |H[k,u']|^2 + n0 )

    参数:
      H:   shape = (K, U),  H[k,u] 表示子载波k, 用户u 的信道增益。
      P:   shape = (K, U),  P[k,u] 表示在子载波k 为用户u 分配的功率。
      n0:  噪声功率 (标量)。
      max_iter:    最大迭代次数。
      tol:         收敛阈值。

    返回:
      a_opt:   shape = (K, U), 优化得到的子载波分配系数。
      obj_vals: list, 每次迭代的目标函数值(近似或真实)。
    """
    K, U = H.shape

    # ---------- 初始化 a^{(0)} ----------
    a_current = np.random.rand(K, U)  # 随机初始化 a
    a_current = np.clip(a_current, 0, 1)  # 保证在 [0, 1] 范围内
    obj_vals = []

    for t in range(max_iter):
        # -----------------------------
        # 计算 S_{k,u} 和 I_{k,u} 的当前值
        # -----------------------------
        S_val = np.zeros((K, U))
        I_val = np.zeros((K, U))

        for k in range(K):
            for u in range(U):
                S_sum = n0
                I_sum = n0
                for up in range(U):
                    S_sum += a_current[k, up] * P[k, u] * H[k, up]
                    if up != u:  # 干扰项
                        I_sum += a_current[k, up] * P[k, u] * H[k, up]
                S_val[k, u] = S_sum
                I_val[k, u] = I_sum

        # -----------------------------
        # 通过线性近似构造目标函数
        # -----------------------------
        a_var = cp.Variable((K, U), nonneg=True)  # 优化变量

        # 线性化目标函数
        obj_expr = 0.0
        for k in range(K):
            for u in range(U):
                # 对 log(S_{k,u}) 和 log(I_{k,u}) 做一阶泰勒展开
                # log(S_{k,u}) 的梯度: 1 / S_{k,u}
                grad_S = 1 / S_val[k, u]  # 梯度
                term_S = grad_S * (cp.sum([a_var[k, up] * P[k, u] * H[k, up] for up in range(U)]) - S_val[k, u])

                # log(I_{k,u}) 的梯度: 1 / I_{k,u}
                grad_I = 1 / I_val[k, u]  # 梯度
                term_I = grad_I * (cp.sum([a_var[k, up] * P[k, u] * H[k, up] for up in range(U) if up != u]) - I_val[k, u])

                # 当前近似的目标项
                obj_expr += term_S - term_I

        # -----------------------------
        # 添加约束
        # -----------------------------
        constraints = [
            a_var >= 0,  # 非负性约束
            a_var <= 1   # 分配系数不超过 1
        ]
        problem = cp.Problem(cp.Maximize(obj_expr), constraints)

        # -----------------------------
        # 求解优化问题
        # -----------------------------
        problem.solve(solver=solver, verbose=verbose)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Solver did not converge at iteration {t}.")
            break

        # 更新 a^{(t+1)}
        a_next = a_var.value

        # -----------------------------
        # 检查收敛性
        # -----------------------------
        diff = np.linalg.norm(a_next - a_current)  # 检查变量变化量
        a_current = a_next.copy()

        # 计算真实的目标函数值
        obj_value = 0.0
        for k in range(K):
            for u in range(U):
                numerator = a_current[k, u] * P[k, u] * H[k, u]
                denominator = 0.0
                for up in range(U):
                    if up != u:
                        denominator += a_current[k, up] * P[k, u] * H[k, up]
                denominator += n0
                gamma_ku = numerator / denominator if denominator > 0 else 0.0
                obj_value += np.log(1.0 + gamma_ku + 1e-15)  # 防止 log(0)
        env_rew = env.cal_sumrate_givenH(a_current.reshape(K, U).transpose().reshape(-1), info['CSI'])[0]
        print(f'itration:{t} obj_value: {obj_value:.4f}, env_value: {env_rew:.3f}')
        obj_vals.append(obj_value)

        # 判断是否满足收敛条件
        if t > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            print(f"Converged at iteration {t}")
            break

    return a_current, obj_vals

a_opt, obj_vals = sca_log_rate_maximization(H, P, n0, solver=cp.GUROBI, max_iter=500, tol=1e-5)
print('seed: ', seed)
print("Optimized allocation a:\n", np.round(a_opt, 2))
print("Objective value history:\n", obj_vals)
print('env: ', env.cal_sumrate_givenH(a_opt.reshape(K, U).transpose(), info['CSI'])[0])
print('done')
