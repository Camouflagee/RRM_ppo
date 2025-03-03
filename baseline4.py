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
seed = np.random.randint(low=0, high=99)
np.random.seed(seed)  # 固定随机种子，保证结果可重复
P_constant = env.BSs[0].Transmit_Power()
P = np.ones((K, U)) * P_constant
H_uk = 10 ** (info['CSI'] / 10)  # info['CSI']: unit dBm
H = (1 / H_uk).reshape(U, K).transpose()


def sca_log_rate_maximization(H, P, n0, solver=cp.MOSEK, max_iter=20, tol=1e-3, verbose=False):
    """
    使用 SCA 方法求解:

        max sum_{k,u} log(1 + gamma_{k,u})
        s.t. 0 <= a_{k,u} <= 1

    其中:
      gamma_{k,u} = (a_{k,u} * P[k,u] * |H[k,u]|^2) /
                    ( sum_{u' != u} a_{k,u'} * P[k,u] * |H[k,u']|^2 + n0 )

    参数:
      H:   shape = (K, U),  H[k,u] 表示子载波k, 用户u 的信道增益(可为复数模平方或其他已知量).
      P:   shape = (K, U),  P[k,u] 表示在子载波k 为用户u 分配的功率(或功率上限).
      n0:  噪声功率 (标量).
      max_iter:    最大迭代次数.
      tol:         收敛阈值(相邻迭代的目标函数值若变化小于 tol, 则停止).

    返回:
      a_opt:   shape = (K, U), 优化得到的子载波分配系数.
      obj_vals: list, 每次迭代的目标函数 (近似或真实) 值.
    """

    K, U = H.shape
    # ---------- 初始化 a^{(0)} ----------
    # 这里给个简单初始化:
    # 可采用均匀分配, 或随机初始化, 或贪婪初始化等
    a_current = np.random.rand(K, U)  # 随机(0,1)
    a_current = np.clip(a_current, 0, 0.5)
    # 均匀分配
    # a_current = np.full((K, U), N_rb / K)
    obj_vals = []
    for t in range(max_iter):
        # -----------------------------
        # 计算干扰项 I_{k,u}(a^{(t)})
        # -----------------------------
        # I_{k,u} = sum_{u' != u} a[k,u'] * P[k,u] * |H[k,u']|^2  +  n0
        I_val = np.zeros((K, U))
        for k in range(K):
            for u in range(U):
                sum_interf = 0.0
                for up in range(U):
                    if up != u:
                        sum_interf += a_current[k, up] * P[k, u] * (H[k, up])
                I_val[k, u] = sum_interf + n0

        # -----------------------------
        # 通过 CVXPY 建模, 求解 a^{(t+1)}
        # -----------------------------
        a_var = cp.Variable((K, U), nonneg=True)

        # 构造 近似目标函数 hat{f}^{(t)}(a)
        # = sum_{k,u} [ log(S_{k,u}(a)) - ( 常数 + 一阶梯度 * (a - a^{(t)}) ) ]
        # 其中 S_{k,u}(a) = n0 + sum_{u'} a[k,u']*P[k,u]*|H[k,u']|^2
        # 做法: 我们只对 -log(I_{k,u}(a)) 做线性化; +log(S_{k,u}(a)) 保持原凹形式(用增广函数或再做近似).
        # 在实际实现里, 可以把 log(S_{k,u}(a)) 也做适当处理, 这里我们直接使用 CP 包内的 log(·).

        # 注意: CVXPY 不直接支持 maximization of sum_of_logs(...) with non-affine arguments
        # 但我们可以手动写一层变换或者利用 DCP 变换技巧(如使用 cp.log(cp.sum(...)))
        # 这里演示一种简化思路：对 log(S_{k,u}) 保留，-log(I_{k,u}) 用线性近似

        # --- 首先准备表达式: log(S_{k,u}(a)) 的和 ---
        sum_log_S = 0.0
        for k in range(K):
            for u in range(U):
                S_expr = n0
                for up in range(U):
                    S_expr += a_var[k, up] * P[k, u] * H[k, up]
                # log(S_expr) -> cp.log( S_expr ), S_expr 要保证 > 0
                sum_log_S += cp.log(S_expr)

        # --- 再准备 -log(I_{k,u}(a^{(t)})) 的一阶近似 ---
        # 常数项 + 线性梯度项
        sum_linear_approx = 0.0
        for k in range(K):
            for u in range(U):
                cst_term = -np.log(I_val[k, u])  # 在 a^{(t)} 的值
                # 梯度 = - 1 / I_val[k,u] * d/d(a[k,u']), 但对 a[k,u'] (u' != u) 才有贡献
                # => gradient_wrt_a[k,u'] = -(P[k,u]*|H[k,u']|^2) / I_val[k,u]
                # a[k,u'] - a_current[k,u'] 是新的变量与旧迭代值之差
                grad_sum = 0.0
                for up in range(U):
                    if up != u:
                        grad_coeff = -(P[k, u] * (H[k, up])) / I_val[k, u]
                        grad_sum += grad_coeff * (a_var[k, up] - a_current[k, up])

                sum_linear_approx += (cst_term + grad_sum)

        # 原目标: sum_{k,u} [ log(S_{k,u}(a)) + ( -log(I_{k,u}(a)) ) ]
        # 近似后变成:  sum_log_S + sum_linear_approx
        # 这里把 sum_linear_approx 中的有效部分“加”进去 (因为是 -log(I_{k,u}) 用的近似)
        obj_expr = sum_log_S + sum_linear_approx

        # 构造约束
        constraints = [
            a_var <= 1,
            a_var >= 0
        ]

        # 定义问题(最大化)
        problem = cp.Problem(cp.Maximize(obj_expr), constraints)

        # 求解
        problem.solve(solver=solver, verbose=verbose)  # 或者选择其他 solver

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Warning: solver did not converge to an optimal solution.")
            break

        # 获得新的解
        a_next = a_var.value

        # -----------------------------
        # 检查收敛性(可选)
        # 这里简单比较 a 的变化量, 或者计算真实的目标函数值
        # -----------------------------
        diff = np.linalg.norm(a_next - a_current)
        a_current = a_next.copy()

        # 计算真实的目标函数值(用于观察收敛)
        # sum_{k,u} log(1 + gamma_{k,u}(a_current))
        obj_value = 0.0
        for k in range(K):
            for u in range(U):
                numerator = a_current[k, u] * P[k, u] * (H[k, u])
                denominator = 0.0
                for up in range(U):
                    if up != u:
                        denominator += a_current[k, up] * P[k, u] * (H[k, up])
                denominator += n0
                gamma_ku = numerator / denominator if denominator > 0 else 0.0
                obj_value += np.log(1.0 + gamma_ku + 1e-15)  # +1e-15 防止log(0)
        env_rew = env.cal_sumrate_givenH(a_current.reshape(K,U).transpose().reshape(-1), info['CSI'])[0]
        print(f'itration:{t} obj_value: {obj_value*BW/10**6:.2f}, env_value: {env_rew:.2f}')
        obj_vals.append(obj_value)

        # 判断是否满足收敛
        if t > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            print(f"Converged at iteration: {t}")
            break

        # 解的变化来判断是否满足收敛
        # if t > 0 and abs(diff) < tol:
        #     print(f"Converged at iteration: {t}")
        #     break
    return a_current, obj_vals


a_opt, obj_vals = sca_log_rate_maximization(H, P, n0, solver=cp.MOSEK, max_iter=500, tol=1e-5)
print("Optimized allocation a:\n", np.round(a_opt, 2))
print("Objective value history:\n", obj_vals)
print('env: ', env.cal_sumrate_givenH(a_opt.reshape(K,U).transpose(), info['CSI'])[0])
print("Optimized allocation a:\n", a_opt)
print('done')
