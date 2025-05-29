import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit

from environmentSB3 import MMSequenceDecisionAdaptiveEnvironmentSB3
from utils import Logger, load_env


class ArgmaxMLP(nn.Module):
    def __init__(self, n, input_dim):
        super().__init__()
        self.n = n

        # 第一层：提取前n个元素的最大值特征
        self.layer1 = nn.Linear(input_dim, n)
        self._init_layer1_weights()

        # 第二层（可选）：恒等映射层，仅用于扩展网络深度
        self.layer2 = nn.Linear(n, n)
        self._init_layer2_weights()

        # 第三层：生成one-hot输出
        self.layer3 = nn.Linear(n, n)
        self._init_layer3_weights()

    def _init_layer1_weights(self):
        """手动初始化第一层权重：隔离前n个元素并计算相对优势"""
        with torch.no_grad():
            # 直接修改Parameter的data属性（而不是替换Parameter）
            for i in range(self.n):
                # 前n个元素中，第i个位置为n，其他为-1；后n个元素为0
                self.layer1.weight.data[i, :self.n] = -1.0
                self.layer1.weight.data[i, i] = self.n-1
                self.layer1.weight.data[i, self.n:] = 0.0
            self.layer1.bias.data.zero_()  # 偏置为0

    def _init_layer2_weights(self):
        """初始化第二层为恒等映射（可选）"""
        with torch.no_grad():
            self.layer2.weight.data = torch.eye(self.n)  # 修改.data属性
            self.layer2.bias.data.zero_()

    def _init_layer3_weights(self):
        """初始化第三层权重：大权重和对角矩阵，偏置为 -w/2"""
        with torch.no_grad():
            w = 1e10  # 权重极大值
            self.layer3.weight.data = torch.eye(self.n) * w
            self.layer3.bias.data = torch.ones(self.n) * (-w / 2)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        h = self.layer2(h)  # 若需移除第二层，注释此行
        y = torch.sigmoid(self.layer3(h))
        return y


# 测试代码
if __name__ == "__main__":
    # n = 3
    # model = ArgmaxMLP(n)
    # model.eval()  # 固定权重
    #
    # # 测试样例1: 输入前n个元素为 [3,5,2], 后n个元素被忽略
    # x_test = torch.tensor([[3.0, 5.0, 2.0, 7.0, 4.0, 6.0]])
    # y_pred = model(x_test)
    # print("输入:", x_test.numpy())
    # print("输出:", y_pred.detach().numpy().round(2))  # 应近似 [0,1,0]
    #
    # # 测试样例2: 输入前n个元素为 [8,1,1], 后n个元素不影响结果
    # x_test = torch.tensor([[8.0, 1.0, 1.0, 99.0, 99.0, 99.0]])
    # y_pred = model(x_test)
    # print("\n输入:", x_test.numpy())
    # print("输出:", y_pred.detach().numpy().round(2))  # 应近似 [1,0,0]
    """评估单个模型的性能"""
    nUE = 12
    nRB = 30
    Nrb = 15
    episode_length = nUE * Nrb
    res = []
    num_pair = []
    test_num = 10

    # 加载环境和模型
    unwrapped_env = load_env(f'Old_experiment_result/seqPPOcons_R2A3_sideinfo/UE{nUE}RB{nRB}/ENV/env.zip')
    n = unwrapped_env.action_space.n
    input_dim = unwrapped_env.observation_space.shape[0]
    model = ArgmaxMLP(n, input_dim)

    # 设置环境参数
    unwrapped_env.error_percent = 0
    unwrapped_env.use_sideinfo = 0
    unwrapped_env.eval_mode = True
    env_class=MMSequenceDecisionAdaptiveEnvironmentSB3
    if not isinstance(unwrapped_env, env_class):
        init_env = env_class(unwrapped_env.sce)
        init_env.__setstate__(unwrapped_env.__getstate__())
        unwrapped_env = init_env

    test_env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)
    # 测试循环
    for _ in range(test_num):
        obs, _ = test_env.reset()
        # obs = torch.tensor(obs, dtype=torch.float64)
        test_env.env.eval_mode = True
        truncated = False
        ob=[obs[:nUE*nRB]]
        while not truncated:
            action = obs[:nUE*nRB].argmax()
            obs, reward, terminated, truncated, info = test_env.step(action)
            ob.append(obs[:nUE*nRB])
            if truncated:
                res.append(reward)
                num_pair.append(sum(obs[nUE * nRB:-1]))

    results = [[], []]
    best_results = [[], []]
    reward, pairs = np.mean(res), np.mean(num_pair)
    print(f"平均奖励: {reward:.3f}")
    print(f"平均配对数: {pairs:.3f}")
