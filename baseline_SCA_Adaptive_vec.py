import copy
import time

import cvxpy as cp
import numpy as np
import yaml
from cvxpy import multiply
from sympy import false

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
'''
there are three kinds of way to add noise on H
1. add noise to H (unit db) with noise of normal distribution with scale np.max(np.abs(H)) # this setting has the marker letter 'A' shown in the experiment record folder name
2. add noise to H (unit db) with noise of normal distribution with scale np.abs(H) # this setting has the marker letter 'A2'
all ways above has the the magnitude issue that (it leads to that the convergence issue in SCA baseline due to the huge magnitude difference of elements)
3. add noise to H (unit real number) with noise of normal distribution with scale np.abs(H)
'''

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


import numpy as np
import cvxpy as cp


def sca_log_rate_maximization_vec(init_a, H, P, n0, solver=cp.MOSEK, max_iter=100, tol=1e-3, verbose=False):
    K, U = H.shape
    a_current = init_a.copy()
    obj_vals = []
    flag = False
    for t in range(max_iter):
        # Vectorized computation of I_val
        I_val = np.zeros((K, U))
        for k in range(K):
            a_k = a_current[k]
            H_k = H[k]
            P_k = P[k]
            sum_all = np.sum(a_k * H_k * P_k)
            sum_without_u = sum_all - a_k * H_k * P_k
            I_val[k] = sum_without_u + n0

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
            grad_coeff = -np.outer(P_k, H_k) / (I_k[:, None]+1e-15)
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
            flag = True
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


is_H_estimated = True
testnum = 10
sol_sce_dict = {}
t1 = time.time()
for idx, (nUE, nRB) in enumerate(
        zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    if idx != 2:
        continue

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
    N_rb = nRB // 2

    # 因为集合 A（基站索引）只有一个元素，所以我们只考虑该基站
    # 生成示例参数：功率 P_{b,k,u} 和信道增益 ||H_{b,k,u}||^2（这里直接用正数表示）
    # seed = np.random.randint(low=0, high=99)
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((K, U)) * P_constant

    _error_percent_list = np.arange(0, 35, 5)/100 if is_H_estimated else [0]
    for _error_percent in _error_percent_list:
        sol_list = []
        obj_list = []
        print("=" * 10, f"error_percent: {_error_percent:.2f}", "=" * 10)
        error_percent = _error_percent
        for test_idx in range(testnum):
            obs, info = env.reset_onlyforbaseline()
            H_dB = info['CSI']  # info['CSI']: unit dBm
            H_uk = 10 ** (H_dB / 10)
            if is_H_estimated:
                # H_error_dB = env.get_estimated_H(H_dB, _error_percent)  # add 5% estimated error
                # H_error_uk = 10 ** (H_error_dB / 10)
                # H_error = (1 / H_error_uk).reshape(U, K).transpose()
                # H_norm_sq = H_error  # H_norm_sq is used by algorithm
                H_error_uk = env.get_estimated_H(H_uk, _error_percent)  # add 5% estimated error
                H_error = (1 / H_error_uk).reshape(U, K).transpose()
                H_norm_sq = H_error  # H_norm_sq is used by algorithm
            else:
                H = (1 / H_uk).reshape(U, K).transpose()
                H_norm_sq = H  # H_norm_sq is used by algorithm

            a_init = np.random.rand(K, U)  # 随机(0,1)
            a, _ = sca_log_rate_maximization_vec(a_init, H_norm_sq, P, n0, solver=cp.MOSEK, max_iter=100, tol=1e-3, verbose=False)
            a_opt_discret = copy.deepcopy(a)
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
                    'H_error': H_norm_sq,
                    'K': K,
                    'U': U,
                    'obj_discrete': obj_discrete
                }
            )
            obj_list.append(obj_discrete)
        cnt_pair = 0
        for sol in sol_list:
            cnt_pair += sum(sum(sol['sol']))
        cnt_pair_avg = cnt_pair / len(sol_list)
        print(f"{testnum}次实验平均后离散化目标函数值:", np.mean(obj_list))
        print(f"{testnum}次实验平均后问题解pair数量:", cnt_pair_avg)

        sol_sce_dict.update(
            {
                f'u{nUE}r{nRB}_err{error_percent}': sol_list
            }
        )
t2 = time.time()

print(f'all test are done, time: {t2 - t1:.2f}s')
