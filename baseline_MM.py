import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic, Logger

"""
1. we use MM algorithm to get the raw solution but not satisfying the constraint
1.1 the surrogate function is to be announced
2. use the discrete projection to get the solution satisfying the constraint
"""


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


testnum = 10
sol_sce_dict = {}
is_H_estimated = True

for idx, (nUE, nRB) in enumerate(
        zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    if idx in [0, 1, 3]:
        continue

    # ============================
    # 1. 参数设置
    # ============================
    init_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')

    env = init_env
    # 设定资源块数目 K, 用户数 U
    K = nRB  # 例如：3个资源块
    U = nUE  # 例如：4个用户
    BW = env.sce.BW
    # 噪声功率 n0 和每用户的资源约束 N_rb
    n0 = env.get_n0()  # 噪声功率
    N_rb = nRB // 2  # 每个用户在所有资源块上分配量之和上限
    P_constant = env.BSs[0].Transmit_Power()
    P = np.ones((K, U)) * P_constant
    print("=" * 10, f"UE{nUE}RB{nRB}场景_Nrb{N_rb}", "=" * 10)
    _error_percent_list = np.arange(0, 65, 5)/100 if is_H_estimated else [0]
    for _error_percent in _error_percent_list:
        error_percent = _error_percent
        sol_list = []
        obj_list = []
        for test_idx in range(testnum):
            obs, info = env.reset_onlyforbaseline()
            # CSI info pre-process (if adding noise)
            H_dB = info['CSI']
            H_uk = 10 ** (H_dB / 10)  # info['CSI']: unit dBm
            if is_H_estimated:
                # H_error_dB = env.get_estimated_H(H_dB, error_percent)  # add 5% estimated error
                # H_error_uk = 10 ** (H_error_dB / 10)
                # H_error = (1 / H_error_uk).reshape(U, K).transpose()
                # H_sq = H_error
                H_error_uk = env.get_estimated_H(H_uk, _error_percent)  # add 5% estimated error
                H_error = (1 / H_error_uk).reshape(U, K).transpose()
                H_sq = H_error  # H_norm_sq is used by algorithm
            else:
                H = (1 / H_uk).reshape(U, K).transpose()
                H_sq = H

            a_init = np.random.rand(K, U)  # 随机(0,1)
            a = a_init
            # 参数设置
            Nk = K  # 资源块数
            Nu = U  # 用户数
            max_iter = 100  # 最大迭代次数
            tol = 1e-4  # 收敛容忍度

            for iter in range(max_iter):
                # 计算当前 gamma 和 A
                gamma = np.zeros((Nk, Nu))
                A = np.zeros((Nk, Nu))
                for k in range(Nk):
                    for u in range(Nu):
                        c_ku = P[k, u] * H_sq[k, u]
                        # 计算干扰项
                        interference = 0
                        for uprime in range(Nu):
                            if uprime != u:
                                d_ku_prime = P[k, uprime] * H_sq[k, uprime]
                                interference += a[k, uprime] * d_ku_prime
                        denominator = interference + n0
                        gamma_ku = (a[k, u] * c_ku) / denominator if denominator != 0 else 0
                        gamma[k, u] = gamma_ku
                        A_ku = a[k, u] * c_ku + interference + n0
                        A[k, u] = A_ku

                # 计算系数
                coeff = np.zeros((Nk, Nu))
                for k in range(Nk):
                    for u in range(Nu):
                        c_ku = P[k, u] * H_sq[k, u]
                        term1 = c_ku / A[k, u] if A[k, u] != 0 else 0
                        term2 = 0
                        for uprime in range(Nu):
                            if uprime != u:
                                gamma_ku_prime = gamma[k, uprime]
                                d_ku_prime = P[k, uprime] * H_sq[k, uprime]
                                A_ku_prime = A[k, uprime]
                                term2 += (gamma_ku_prime * d_ku_prime) / A_ku_prime if A_ku_prime != 0 else 0
                        coeff[k, u] = term1 - term2

                # 更新a，根据系数决定0或1
                a_new = np.where(coeff > 0, 1, 0)

                # 检查收敛
                if np.max(np.abs(a_new - a)) < tol:
                    # print(f"收敛于第 {iter} 次迭代")
                    break
                a = a_new.copy()

            # print("优化后的 a 矩阵：")
            # print(a)

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
        cnt_pair = 0
        for sol in sol_list:
            cnt_pair += sum(sum(sol['sol']))
        cnt_pair_avg = cnt_pair / len(sol_list)
        if is_H_estimated:
            print(f'error_rate:{error_percent:.3f}')
        print(f"{testnum}次实验平均后离散化目标函数值:{np.mean(obj_list):.3f}" )
        print(f"{testnum}次实验平均后问题解pair数量:{cnt_pair_avg:.1f}" )

        sol_sce_dict.update(
            {
                f'u{nUE}r{nRB}_err{error_percent}': sol_list
            }
        )
print('all test are done')
