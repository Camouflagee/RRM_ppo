import copy
import os
import pickle
import time

import numpy as np
import cvxpy as cp
from gymnasium.wrappers import TimeLimit

from environmentSB3 import SequenceDecisionAdaptiveEnvironmentSB3
from module.sequence_ppo import SequencePPO
from utils import load_env, get_TimeLogEvalDir


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


def GradProj(init_a, H_uk, N_rb, K, U, P, n0, BW, eta=0.06, max_iter=100, tol=1e-4, verbose=False, solver=None):
    # 梯度上升参数
    K, U = init_a.shape
    H_norm_sq = H_uk
    # -------------------------------
    # 初始化 a_{k,u} 满足约束: a在[0,1]内，且每个用户对所有资源块的分配和不超过 N_rb
    # 初始化时可以均匀分配或随机初始化
    a = init_a
    for u in range(U):
        if a[:, u].sum() > N_rb:
            a[:, u] = a[:, u] * (N_rb / a[:, u].sum())

    def projection(a_u, N_rb):
        # the projection used in the algorithm
        return continue_projection(a_u, N_rb)

    # 计算目标函数值
    def compute_rate(a, P, _H, n0, _user_burst_mat=None):
        rate = 0
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
    for iter in range(max_iter):
        current_rate = compute_rate(a, P, H_norm_sq, n0)
        # rate_history.append(current_rate * BW // 10 ** 6)

        grad = compute_grad(a, P, H_norm_sq, n0)
        a_new = a + eta * grad

        # 对每个用户分别投影到可行域
        for u in range(U):
            a_new[:, u] = projection(a_new[:, u], N_rb)

        # 终止条件：若目标函数增量很小则退出
        if np.abs(compute_rate(a_new, P, H_norm_sq, n0) - current_rate) < tol:
            a = a_new
            if verbose:
                print(f"Converged at iter {iter}")
            break
        a = a_new
    return a, None


def MM(init_a, H_uk, N_rb, K, U, P, n0, BW, eta=0.06, max_iter=100, tol=1e-4, verbose=False, solver=None):
    a = init_a
    # 参数设置
    Nk = K  # 资源块数
    Nu = U  # 用户数
    max_iter = max_iter  # 最大迭代次数
    H_sq = H_uk
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
            if verbose:
                print(f"收敛于第 {iter} 次迭代")
            break
        a = a_new.copy()
    return a_new, None

def SCA_vec(init_a, H_uk, N_rb, K, U, P, n0, BW, eta=0.06, max_iter=100, tol=1e-4, verbose=False, solver=cp.MOSEK):
    H = H_uk
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
                print(f"SCA Converged at iteration {t}")
            break
    if flag:
        print('SCA Warning: Solver did not converge.')
    return a_current, obj_vals

def get_model_paths(root_dir):
    """获取所有训练好的模型路径"""
    model_paths = []
    error_rates = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.zip') and 'model_saves' in root:
                # if file.endswith('.zip') and 'model_saves' in root and 'best' in file:
                model_paths.append(os.path.join(root, file))
                # 从路径中提取error rate
                error_match = re.search(r'error_([0-9.]+)', root)
                error_rate = float(error_match.group(1)) if error_match else 0.0
                error_rates.append(error_rate)
    return model_paths, error_rates

def eval_model(model_path, error_rate, use_sideinfo, given_obs=None):
    """评估单个模型的性能"""
    nUE = 12
    nRB = 30
    Nrb = 15
    episode_length = nUE * Nrb
    res = []
    num_pair = []
    test_num = 80

    # 加载环境和模型
    unwrapped_env = load_env(f'Experiment_result/seqPPOcons_R2A3_sideinfo/UE{nUE}RB{nRB}/ENV/env.zip')
    model = SequencePPO.load(model_path)

    # 设置环境参数
    unwrapped_env.error_percent = error_rate
    unwrapped_env.use_sideinfo = use_sideinfo
    unwrapped_env.eval_mode = True
    assert isinstance(unwrapped_env,SequenceDecisionAdaptiveEnvironmentSB3)
    test_env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)

    # 测试循环
    for _ in range(test_num):
        obs, _ = test_env.reset_onlyforbaseline(given_obs,)
        truncated = False
        while not truncated:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if truncated:
                res.append(reward)
                num_pair.append(sum(obs[nUE * nRB:]))

    return np.mean(res), np.mean(num_pair)


def evaluate_models(logger, model_dir, use_sideinfo, given_obs=None):
    """评估指定目录下的所有模型"""
    model_paths, error_rates = get_model_paths(model_dir)
    results = [[], []]
    best_results = [[], []]

    for idx, (model_path, error_rate) in enumerate(zip(model_paths, error_rates)):
        reward, pairs = eval_model(model_path, error_rate, use_sideinfo, given_obs)
        logger.log(f"\n错误率: {error_rate:.2f}")
        logger.log(f"平均奖励: {reward:.3f}")
        logger.log(f"平均配对数: {pairs:.3f}")
        results[0].append(reward)
        results[1].append(pairs)
        if idx % 2 == 0:
            best_results[0].append(max(results[0][-2:]))
            best_results[1].append(max(results[1][-2:]))

    return best_results, results
if __name__ == '__main__':
    is_H_estimated = True
    testnum = 10
    t1 = time.time()
    for idx, (nUE, nRB) in enumerate(
            zip([5, 10, 12, 15], [10, 20, 30, 40])):  # 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
        if idx != 2:
            continue

        print("\n","=" * 10, f"UE{nUE}RB{nRB}场景", "=" * 10)
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

        error_percent_list = np.arange(0, 65, 5) / 100 if is_H_estimated else [0]
        # generate the H instance for experiment
        H_list = []
        # for test_idx in range(testnum):
        #     obs, info = env.reset_onlyforbaseline()
        #     H_list.append((obs, info))

        obs_path='Experiment_result/seqPPOcons_R2A3_fixobs_mm/UE12RB30/E1_Nrb15_epl_180_error_0.00/date20250425time140309/obsinfo.pkl'
        with open(obs_path, 'rb') as f:
            loaded_data = pickle.load(f)
        obs, info = loaded_data['obs'], loaded_data['info']
        H_list.append((obs, info))

        def run_exp(_H_list, _error_percent_list, algo):
            sol_sce_dict = {}
            mean_cnt_per_error=[]
            mean_obj_per_error=[]
            for eidx, _error_percent in enumerate(_error_percent_list):
                sol_list = []
                obj_list = []
                print("=" * 10, f"error_percent: {_error_percent:.2f}", "=" * 10)
                error_percent = _error_percent
                for obs, info in H_list:
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

                    a, _ = algo(a_init, H_norm_sq, N_rb, K, U, P, n0, BW, eta=0.06, max_iter=100, tol=1e-4, verbose=False,
                                  solver=cp.MOSEK)
                    a_opt_discret = copy.deepcopy(a)
                    for u in range(U):
                        a_opt_discret[:, u] = discrete_project_per_user(a_opt_discret[:, u], N_rb)

                    obj_discrete = env.cal_sumrate_givenH(a_opt_discret.reshape(K, U).transpose(), info['CSI'])[0]
                    sol_list.append(
                        {
                            'sol': a_opt_discret,
                            'obj_discrete': obj_discrete,
                            'H': info['CSI'],
                            'H_error': H_norm_sq,
                        }
                    )
                    obj_list.append(obj_discrete)
                cnt_pair = 0
                for sol in sol_list:
                    cnt_pair += sum(sum(sol['sol']))
                cnt_pair_avg = cnt_pair / len(sol_list)
                print(f"{testnum}次实验平均后离散化目标函数值:", np.mean(obj_list))
                print(f"{testnum}次实验平均后问题解pair数量:", cnt_pair_avg)
                mean_obj_per_error.append(np.round(np.mean(obj_list),3))
                mean_cnt_per_error.append(np.round(cnt_pair_avg,3))
                sol_sce_dict.update(
                    {
                        f'u{nUE}r{nRB}_err{error_percent}': sol_list
                    }
                )
            info = (mean_cnt_per_error, mean_obj_per_error)
            print(info)
            return sol_sce_dict, info
        res={}
        for idx, (name, algo) in enumerate(zip(['MM','GradProj','SCA'],[MM,GradProj,SCA_vec])):
            # if idx!=2 :
            #     continue
            print("*"*20,f"{name} experiment","*"*20)
            sol_sce_dict, info = run_exp(H_list, error_percent_list, algo)
            res.update({'name':info})
    t2 = time.time()

    print(f'all test are done, time: {t2 - t1:.2f}s')

    # plot
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')
    # 设置支持中文的字体（Windows系统通常使用SimHei，Mac使用PingFang SC）
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei', 'PingFang SC', 'Heiti TC']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 常见中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建数据
    x = np.arange(0, 65, 5) / 100  # 横轴数据

    # 五组不同的曲线函数
    # y1 = np.sin(x)  # SeqPPO side info
    # y2 = np.cos(x)  # SeqPPO no side info
    # y3 = np.exp(-x / 5)  # MM
    # y4 = np.log(x + 1)  # SCA
    # y5 = [70.80, 70.97, 70.68, 70.69, 70.60, 70.55, 70.56, 70.49, 70.45, 70.41, 70.29, 70.16, 70.10]  # GradProj

    # 创建图形
    plt.figure(figsize=(10, 6))  # 设置图形大小
    color=['blue','red','green','purple','orange']
    dot=['','--',':','-.']
    for d,clr,algo_name in enumerate(zip(color,dot,['MM','GradProj','SCA'])):
        y = info[algo_name][1]
        plt.plot(x, y, label=algo_name, color=clr)
    log_dir=get_TimeLogEvalDir(model_name='baseline', args='UE12RB30')
    fig_path=os.path.join(log_dir,'figures.jpg')
    # 绘制五组曲线
    # plt.plot(x, y1, label='SeqPPO', color='blue')
    # plt.plot(x, y2, label='cos(x)', color='red', linestyle='--')
    # plt.plot(x, y3, label='exp(-x/5)', color='green', linestyle=':')
    # plt.plot(x, y4, label='log(x+1)', color='purple', linestyle='-.')
    # plt.plot(x, y5, label='SCA', color='orange')

    # 添加图例
    plt.legend(loc='upper right')

    # 添加轴标签
    plt.xlabel('误差率', fontsize=12)
    plt.ylabel('系统总和速率(Mbps)', fontsize=12)

    # 添加标题
    plt.title('在信道', fontsize=14)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    plt.show()
