import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic, Logger

# with open('config/config_environment_setting.yaml', 'r') as file:
#     env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
# sce = env_args
#
# NonSequencedEnv = load_env('saved_env/BS1UE20/env.zip')
# init_env = SequenceDecisionEnvironmentSB3(env_args)
# init_env.__setstate__(NonSequencedEnv.__getstate__())
# def get_env_reward(env, a, H):
#     K = env.sce.nRBs
#     U = env.sce.nUEs
#     return env.cal_sumrate_givenH(a.reshape(K, U).transpose(), H)
"""
1. we use SCA to get the raw solution but not satisfying the constraint
2. use the discrete projection to get the solution satisfying the constraint
"""

# ============================
# 1. 参数设置
# ============================

# env = init_env
# # 设定资源块数目 K, 用户数 U
# K = env.nRB  # 例如：3个资源块
# U = env.nUE  # 例如：4个用户
# BW = env.sce.BW
# # 噪声功率 n0 和每用户的资源约束 N_rb
# n0 = env.get_n0()  # 噪声功率
# N_rb = env.sce.rbg_Nb  # 每个用户在所有资源块上分配量之和上限
# obs, info = env.reset_onlyforbaseline()
#
# # 因为集合 A（基站索引）只有一个元素，所以我们只考虑该基站
# # 生成示例参数：功率 P_{b,k,u} 和信道增益 ||H_{b,k,u}||^2（这里直接用正数表示）
# seed = np.random.randint(low=0, high=99)
# np.random.seed(seed)  # 固定随机种子，保证结果可重复
# P_constant = env.BSs[0].Transmit_Power()
# P = np.ones((K, U)) * P_constant
# H_uk = 10 ** (info['CSI'] / 10)  # info['CSI']: unit dBm
# H = (1 / H_uk).reshape(U, K).transpose()


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


def sca_log_rate_maximization(init_a, H, P, n0, solver=cp.MOSEK, max_iter=100, tol=1e-3, verbose=False):
    """
    使用 SCA 方法求解:

        max sum_{k,u} log(1 + gamma_{k,u})
        s.t. 0 <= a_{k,u} <= 1

    其中:
      gamma_{k,u} = (a_{k,u} * P[k,u] * |H[k,u]|^2) /
                    ( sum_{u' != u} a_{k,u'} * P[k,u] * |H[k,u']|^2 + n0 )

    参数:
      H:   shape = (K, U),  H[k,u] 表示子载波k, 用户u 的信道增益, 注无需平方计算, H已经平方计算过了.
      P:   shape = (K, U),  P[k,u] 表示在子载波k 为用户u 分配的功率(或功率上限).
      n0:  噪声功率 (标量).
      max_iter:    最大迭代次数.
      tol:         收敛阈值(相邻迭代的目标函数值若变化小于 tol, 则停止).

    返回:
      a_opt:   shape = (K, U), 优化得到的子载波分配系数.
      obj_vals: list, 每次迭代的目标函数 (近似或真实) 值.
    """

    K, U = H.shape
    a_current = init_a

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

        def cal_sumrate_local(a_current, H, _K, _U):
            """

            :param a_current: shape(K,U)
            :param H: shape(K,U), K is RB index, U is user index.
            :return:
            """
            # 计算真实的目标函数值(用于观察收敛)
            # sum_{k,u} log(1 + gamma_{k,u}(a_current))
            obj_val = 0.0
            for k in range(_K):
                for u in range(_U):
                    numerator = a_current[k, u] * P[k, u] * (H[k, u])
                    denominator = 0.0
                    for up in range(U):
                        if up != u:
                            denominator += a_current[k, up] * P[k, u] * (H[k, up])
                    denominator += n0
                    gamma_ku = numerator / denominator if denominator > 0 else 0.0
                    obj_val += np.log(1.0 + gamma_ku + 1e-15)  # +1e-15 防止log(0)
            return obj_val

        obj_value = cal_sumrate_local(a_current, H, _K=K, _U=U)
        env_rew = env.cal_sumrate_givenH(a_current.reshape(K, U).transpose().reshape(-1), info['CSI'])[0]
        if verbose:
            print(f'itration:{t} obj_value: {obj_value * BW / 10 ** 6:.2f}, env_value: {env_rew:.2f}')
        obj_vals.append(obj_value)

        # 判断是否满足收敛
        if t > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            if verbose:
                print(f"Converged at iteration: {t}")
            break

        # 解的变化来判断是否满足收敛
        # if t > 0 and abs(diff) < tol:
        #     print(f"Converged at iteration: {t}")
        #     break
    return a_current, obj_vals
def sca_log_rate_maximization_vec(init_a, H, P, n0, solver=cp.MOSEK, max_iter=100, tol=1e-3, verbose=False):
    K, U = H.shape
    a_current = init_a.copy()
    obj_vals = []
    flag=False
    for t in range(max_iter):
        # Vectorized computation of I_val
        I_val = np.zeros((K, U))
        for k in range(K):
            a_k = a_current[k]
            H_k = H[k]
            P_k = P[k]
            sum_all = np.sum(a_k * H_k)
            sum_without_u = sum_all - a_k * H_k
            I_val[k] = sum_without_u * P_k + n0

        # CVXPY problem setup with vectorized operations
        a_var = cp.Variable((K, U), nonneg=True)
        sum_log_S = 0
        for k in range(K):
            P_k = P[k, :]
            H_k = H[k, :]
            # Construct P_k_H matrix (U, U) where each row u is P_k[u] * H_k
            P_k_H = cp.outer(P_k, H_k)
            sum_terms = P_k_H @ a_var[k, :]
            S_k = n0 + sum_terms
            sum_log_S += cp.sum(cp.log(S_k))

        # Vectorized computation of sum_linear_approx
        sum_cst_terms = -np.sum(np.log(I_val))
        sum_grad_terms = 0
        for k in range(K):
            P_k = P[k, :]
            H_k = H[k, :]
            I_k = I_val[k, :]
            # Compute gradient coefficients matrix (U, U)
            grad_coeff = -np.outer(P_k, H_k) / I_k[:, None]
            np.fill_diagonal(grad_coeff, 0)
            a_diff = a_var[k, :] - a_current[k, :]
            sum_grad_terms += cp.sum(grad_coeff @ a_diff)
        sum_linear_approx = sum_cst_terms + sum_grad_terms

        # Objective and constraints
        obj_expr = sum_log_S + sum_linear_approx
        constraints = [a_var <= 1, a_var >= 0]
        problem = cp.Problem(cp.Maximize(obj_expr), constraints)
        problem.solve(solver=solver, verbose=verbose)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            flag=True
            break

        a_next = a_var.value
        a_current = a_next.copy()

        # Vectorized objective calculation
        gamma = np.zeros((K, U))
        for k in range(K):
            a_k = a_current[k]
            P_k = P[k]
            H_k = H[k]
            sum_matrix = a_k[None, :] * P_k[:, None] * H_k[None, :]
            sum_all = np.sum(sum_matrix, axis=1)
            numerator = np.diag(sum_matrix)
            denominator = sum_all - numerator + n0
            gamma[k] = numerator / (denominator + 1e-15)
        obj_val = np.sum(np.log(1 + gamma))
        obj_vals.append(obj_val)

        if t > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break
    if flag:
        print('Warning: Solver did not converge.')
    return a_current, obj_vals

is_H_estimated=True
testnum = 5
sol_sce_dict = {}
for idx, (nUE, nRB) in enumerate(
        zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    if idx != 2:
        continue
    sol_list = []
    obj_list = []
    print("=" * 10, f"UE{nUE}RB{nRB}场景", "=" * 10)
    # logger = Logger(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/baseline_output.txt')
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
    N_rb = nRB//2

    # 因为集合 A（基站索引）只有一个元素，所以我们只考虑该基站
    # 生成示例参数：功率 P_{b,k,u} 和信道增益 ||H_{b,k,u}||^2（这里直接用正数表示）
    # seed = np.random.randint(low=0, high=99)
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((K, U)) * P_constant

    _error_percent_list = np.arange(0, 0.65, 0.05) if is_H_estimated else [0]
    for _error_percent in _error_percent_list:
        print("=" * 10, f"error_percent: {_error_percent:.2f}", "=" * 10)
        error_percent = _error_percent
        for test_idx in range(testnum):
            obs, info = env.reset_onlyforbaseline()
            H_dB = info['CSI']# info['CSI']: unit dBm
            H_uk = 10 ** (H_dB / 10)
            H = (1 / H_uk).reshape(U, K).transpose()
            if is_H_estimated:
                # H_error_dB = env.get_estimated_H(H_dB, _error_percent)  # add 5% estimated error
                # H_error_uk = 10 ** (H_error_dB / 10)
                # H_error = (1 / H_error_uk).reshape(U, K).transpose()
                # H_norm_sq = H_error  # This H is used by algorithm
                H_error_uk = env.get_estimated_H(H_uk, _error_percent)  # add 5% estimated error
                H_error = (1 / H_error_uk).reshape(U, K).transpose()
                H_norm_sq = H_error  # H_norm_sq is used by algorithm

            else:
                H_norm_sq = H  # This H is used by algorithm

            a_init = np.random.rand(K, U)  # 随机(0,1)
            a, _ = sca_log_rate_maximization_vec(a_init, H_norm_sq, P, n0, solver=cp.MOSEK, max_iter=100, tol=1e-3, verbose=False)
            a_opt_discret = a
            for u in range(U):
                a_opt_discret[:, u] = discrete_project_per_user(a_opt_discret[:, u], N_rb)

            obj_discrete = env.cal_sumrate_givenH(a_opt_discret.reshape(K, U).transpose(), info['CSI'])[0]
            # print("离散映射后最终目标值：", obj_discrete)
            # print("a_opt_discret:")
            # print(np.array2string(a_opt_discret, separator=', '))
            sol_list.append(
                {
                    'sol': a_opt_discret,
                    'H': info['CSI'],
                    'K': K,
                    'U': U,
                    'obj_discrete': obj_discrete
                }
            )
            obj_list.append(obj_discrete)
        cnt_pair=0
        for sol in sol_list:
            cnt_pair += sum(sum(sol['sol']))
        cnt_pair_avg = cnt_pair/len(sol_list)
        print(f"{testnum}次实验平均后离散化目标函数值:", np.mean(obj_list))
        print(f"{testnum}次实验平均后问题解pair数量:", cnt_pair_avg)

        sol_sce_dict.update(
            {
                f'u{nUE}r{nRB}_err{error_percent}': sol_list
            }
        )
print('all test are done')
