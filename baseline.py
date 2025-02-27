import numpy as np
import yaml
from scipy.optimize import minimize

from environmentSB3 import SequenceDecisionEnvironmentSB3
from utils import DotDic, load_env

with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
sce = env_args

Noise = 10 ** (sce.N0 / 10) * sce.BW  # Calculate the noise
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
np.random.seed(42)  # 固定随机种子，保证结果可重复
P = env.BSs[0].Transmit_Power()
H = 10 ** (info['CSI'].reshape(U, K).transpose() / 10) / 10**12 # info['CSI']: unit dBm

# P = 50 # dBm # 每个资源块、每个用户对应的发射功率
# H = np.random.uniform(0.5, 1.5, (K, U))  # 相应的信道“增益”或 ||H||^2
import warnings

# 将警告提升为异常
warnings.simplefilter("error")


# ============================
# 2. 定义目标函数
# ============================
def objective(a):
    # todo check the objective definition
    """
    目标函数：负的和对数函数，即 -sum_{u,k} log(1+SINR_{k,u})
    其中 SINR_{k,u} = (a_{k,u}*P_{k,u}*H_{k,u}) / ( n0 + sum_{u'≠u} a_{k,u'}*P_{k,u'}*H_{k,u'} )
    """
    # 将变量 a 转换为形状为 (K, U) 的矩阵，便于处理每个资源块、每个用户的分配
    a_matrix = a.reshape((K, U))

    total = 0.0
    # 对所有用户 u 与资源块 k 求和
    for u in range(U):
        for k in range(K):
            numerator = a_matrix[k, u] * P * H[k, u]
            # 对于同一固定资源块 k，对于其他用户产生的干扰
            interference = n0 + sum(a_matrix[k, u2] * P / H[k, u2]
                                    for u2 in range(U) if u2 != u)
            SINR = numerator / interference
            rate = np.log(1 + SINR)
            total += rate

    # 注意：使用 minimize, 故返回负值
    return -total


# ============================
# 3. 定义约束与变量边界
# ============================
# 由于题中约束 (cons4): 每个用户 u 在该基站所有资源块分配之和不超过 N_rb
constraints = []
for u1 in range(U):
    # 对于每个用户 u，约束： sum_{k} a_{k,u} <= N_rb  即  N_rb - sum_{k} a_{k,u} >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda a, u=u1: N_rb - np.sum(a.reshape((K, U))[:, u])
    })

# 此外要求 a_{k,u} >= 0
bounds = [(0, 1)] * (K * U)

# ============================
# 4. 求解优化问题
# ============================
# 初始猜测: 每个 a_{k,u} 初值设为 0.1
a0 = np.full((K * U,), 0.1)

# 使用 SLSQP 方法求解（也可以试试 'trust-constr' 等其他方法）
result = minimize(objective, a0, method='L-BFGS-B', bounds=bounds, )
# result = minimize(objective, a0, method='SLSQP', bounds=bounds, constraints=constraints)  # constraints=constraints

# ============================
# 5. 输出结果
# ============================
if result.success:
    a_optimal = result.x.reshape((K, U))
    print("最优解 a^t（每行对应一个资源块，每列对应一个用户）：")
    print(a_optimal)
    optimal_value = -result.fun
    print("\n最优目标函数值（和对数速率）：", optimal_value)
else:
    print("优化没有收敛：", result.message)
print('done')