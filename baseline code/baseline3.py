import cvxpy as cp
import numpy as np
import yaml

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import load_env, DotDic

with open('../config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args

NonSequencedEnv = load_env('../saved_env/BS1UE20/env.zip')
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
np.random.seed(42)  # 固定随机种子，保证结果可重复
P = env.BSs[0].Transmit_Power()
H_uk = 10 ** (info['CSI'] / 10)  # info['CSI']: unit dBm
# =========================
# 参数及数据初始化
# =========================
np.random.seed(0)

# 集合定义
num_users = U  # 用户数量
num_rb = K  # 资源块数量
N_rb = N_rb  # 每个用户最大可分配的资源块数量

# 随机生成功率 P 和信道参数 H（保证为正数），并计算 ||H||^2
# P0 = np.random.uniform(0.5, 2.0, (num_rb, num_users))
P = np.ones((num_rb, num_users)) * P
# H0 = np.random.uniform(0.5, 2.0, (num_rb, num_users))
Hsq = (1/info['CSI']).reshape(U, K).transpose()
n0 = n0

# 初始化变量 a（例如均匀分配），确保初始解在 [0, 1] 内
# a_opt = np.zeros((num_rb, num_users))
# for u in range(num_users):
#     a_opt[:, u] = (N_rb / num_rb) * np.ones(num_rb)
a_opt = np.full((K , U), N_rb / K)

# SCA参数设定
max_iter = 500
tol = 1e-5

# =========================
# SCA 主迭代过程
# =========================
for it in range(max_iter):
    # 定义待优化变量 a（非负）
    a = cp.Variable((num_rb, num_users), nonneg=True)
    obj_terms = []

    # 针对每个资源块 k 和用户 u 构造目标函数项
    for k in range(num_rb):
        for u in range(num_users):
            # 计算干扰项：用户 u 在资源块 k 下的干扰来自于其他用户 u2 (u2 ≠ u)
            interference_expr = 0
            interference_value = 0
            for u2 in range(num_users):
                if u2 != u:
                    interference_expr += a[k, u2] * P[k, u2] * Hsq[k, u2]
                    interference_value += a_opt[k, u2] * P[k, u2] * Hsq[k, u2]

            # 第一项：信号 + 干扰 + 噪声
            total_expr = a[k, u] * P[k, u] * Hsq[k, u] + interference_expr + n0
            term1 = cp.log(total_expr)

            # 第二项：对干扰部分的线性化
            x_opt = interference_value + n0
            term2_approx = cp.log(x_opt) + (1 / x_opt) * ((interference_expr + n0) - x_opt)

            # 添加当前的 surrogate 项
            obj_terms.append(term1 - term2_approx)

    # 构造总目标函数：对所有 k, u 求和
    objective = cp.Maximize(cp.sum(obj_terms))

    # 构造约束条件：
    constraints = []
    # 每个用户 u 的资源块分配之和不超过 N_rb
    for u in range(num_users):
        constraints.append(cp.sum(a[:, u]) <= N_rb)
    # 添加全局变量的上界约束：所有 a 的取值都必须小于等于 1
    constraints.append(a <= 1)

    # 构造并求解凸优化问题
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("在迭代 {} 中，问题求解状态为：{}".format(it, prob.status))
        break

    # 获取并打印当前最优目标函数值
    optimal_value = prob.value
    print("迭代 {}: 当前最优目标函数值 = {:.5f}".format(it, optimal_value*BW/10**6))
    print("env.calsumrate: ", env.cal_sumrate_givenH(a_opt.transpose(), Hsq))

    # 更新变量 a
    a_new = a.value

    # 检查收敛条件（采用欧式范数距离）
    diff = np.linalg.norm(a_new - a_opt)
    print("迭代 {}: 变量更新的范数差 diff = {:.5f}".format(it, diff))

    if diff < tol:
        a_opt = a_new
        break
    a_opt = a_new

print("最终优化得到的分配 a^t_{k,u}：")
print(a_opt)
print("最终最优目标函数值 = {:.5f}".format(optimal_value*BW/10**6))
print("env.calsumrate: ", env.cal_sumrate_givenH(a_opt.transpose(), Hsq))
print('done')
# =========================
# SCA 主迭代过程 vectorizing
# =========================
# 初始化变量 a（例如均匀分配），确保初始解在 [0, 1] 内
a_opt = np.zeros((num_rb, num_users))
for u in range(num_users):
    a_opt[:, u] = (N_rb / num_rb) * np.ones(num_rb)

# SCA参数设定
max_iter = 500
tol = 1e-8

print("="*30, 'vectorizing', "="*30)
for it in range(max_iter):
    # 定义待优化变量 a（非负，且在 [0, 1] 范围内）
    a = cp.Variable((num_rb, num_users), nonneg=True)

    # 计算干扰项 (矩阵形式)
    # 干扰项 I[k, u] 包含每个用户 u 在资源块 k 下的干扰，由其他用户的分配决定
    interference_expr = cp.sum(cp.multiply(a, P * Hsq), axis=1, keepdims=True) - cp.multiply(a, P * Hsq)

    # 信号 + 干扰 + 噪声 (矩阵形式)
    signal_expr = cp.multiply(a, P * Hsq)  # 信号项
    total_expr = signal_expr + interference_expr + n0  # 总项

    # 当前干扰值（用于线性化，使用 a_opt 计算）
    interference_value = np.sum(a_opt * P * Hsq, axis=1, keepdims=True) - (a_opt * P * Hsq)

    # 对干扰项的线性化
    x_opt = interference_value + n0
    term2_approx = cp.log(x_opt) + cp.multiply(1 / x_opt, (interference_expr - interference_value))

    # 构造目标函数：对所有资源块和用户的目标项求和
    objective = cp.Maximize(cp.sum(cp.log(total_expr) - term2_approx))

    # 构造约束条件：
    constraints = []
    # 每个用户 u 的资源块分配之和不超过 N_rb
    constraints.append(cp.sum(a, axis=0) <= N_rb)
    # 添加全局变量的上界约束：所有 a 的取值都必须小于等于 1
    constraints.append(a <= 1)

    # 构造并求解凸优化问题
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("在迭代 {} 中，问题求解状态为：{}".format(it, prob.status))
        break

    # 获取并打印当前最优目标函数值
    optimal_value = prob.value
    print("迭代 {}: 当前最优目标函数值 = {:.5f}".format(it, optimal_value*BW/10**6))
    print("env.calsumrate: ", env.cal_sumrate_givenH(a_opt.transpose(), info['CSI']))
    # 更新变量 a
    a_new = a.value

    # 检查收敛条件（采用欧式范数距离）
    diff = np.linalg.norm(a_new - a_opt)
    print("迭代 {}: 变量更新的范数差 diff = {:.5f}".format(it, diff))

    if diff < tol:
        a_opt = a_new
        break
    a_opt = a_new

print("最终优化得到的分配 a^t_{k,u}：")
print(np.round(a_opt,2))
print("最终最优目标函数值 = {:.5f}".format(optimal_value*BW/10**6))
print("env.calsumrate: ", env.cal_sumrate_givenH(a_opt.transpose(), info['CSI']))
