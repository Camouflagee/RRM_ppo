# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：UARL
@File    ：environment.py
@Author  ：Jiansheng LI
@Date    ：2023/11/27 14:12 
'''
import time
###
# 1. fading
# 2. state how to define a MDP
# agent: BS
# state: [SINR of its own UE, and history information] ?
# action space: control of decision variable [{0,1} * K] where K is (the number of channels or RB indexs)
# reward: utility function of (transmision rate?)
# transition prob matrix:? modules-free just save in buffer
# discounted rate: /gamma decrease or increase?
# 3. when to converge: learning rate; crossentropy of output(action distribution become more determistic)
from random import random, uniform, choice, randrange, randint
import numpy as np
from numpy import pi, sqrt, log10
import torch

import gymnasium as gym

# import sys
# sys.modules["gym"] = gym
# class MyMultiDiscrete(spaces.MultiDiscrete):
#     def __init__(self, num):
#         super().__init__(num)
#         self._masks = [np.arange(n) for n in range(num)]
#
#     def sample(self):
#         vals = []
#         for mask in self._masks:
#             val = np.random.choice(mask)
#             vals.append(val)
#             self._masks = mask[mask != val]  # 更新mask
#         return vals
import random


class BS:  # Define the base station (Agent)

    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm):
        self.sce = sce
        self.id = BS_index
        self.BStype = BS_type
        self.BS_Loc = BS_Loc
        self.BS_Radius = BS_Radius
        self.UE_set = []  # 信号范围内的用户集合
        self.Tx_Power_dBm = Tx_Power_dBm
        # self.Tx_Power = 10 ** (Tx_Power_dBm / 10)
        # def __getstate__(self):

    #     return self.sce,self.id,self.BStype,self.BS_Loc,self.BS_Radius,self.UE_set
    #      return self.__dict__
    # def __setstate__(self, state):
    #     self.sce = state['sce']
    #     self.id = state['id']
    #     self.BStype = state['BStype']
    #     self.BS_Loc = state['BS_Loc']
    #     self.BS_Radius = state['BS_Radius']
    #     self.UE_set = state['UE_set']
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def reset(self):  # Reset the channel status
        self.Ch_State = np.zeros(self.sce.nChannel)

    def Get_Location(self):
        return self.BS_Loc

    def Transmit_Power(self):  # Calculate the transmit power of a BS
        # if self.BStype == "MBS":
        #     Tx_Power_dBm = 40
        # elif self.BStype == "PBS":
        #     Tx_Power_dBm = 30
        # elif self.BStype == "FBS":
        #     Tx_Power_dBm = 20
        return 10 ** (self.Tx_Power_dBm / 10)  # Transmit power in dBm, no consideration of power allocation now

    def Transmit_Power_dBm(self):  # Calculate the transmit power of a BS
        # if self.BStype == "MBS":
        #     Tx_Power_dBm = 40
        # elif self.BStype == "PBS":
        #     Tx_Power_dBm = 30
        # elif self.BStype == "FBS":
        #     Tx_Power_dBm = 20
        return self.Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now

    def Select_Action(self, state, scenario, eps_threshold):  # Select action for a user based on the network state
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels
        sample = random()
        if sample < eps_threshold:  # epsilon-greeedy policy
            with torch.no_grad():
                Q_value = self.model_policy(state)  # Get the Q_value from DNN
                action = Q_value.max(0)[1].view(1, 1)
        else:
            action = torch.tensor([[randrange(L * K)]], dtype=torch.long)
        return action


class UE:  # Define the agent (UE)

    def __init__(self, x, y, index):  # Initialize the agent (UE)
        self.id = index
        self.location = [x, y]
        self.original_location = [x, y]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.original_location = state['location']

    # def Set_Location(self, BS_location, BS_type : str):  # Initialize the location of the agent
    #     #
    #     #     LocM = BS_location
    #     #     Loc_agent = np.zeros(2)  # 预设agent的位置
    #     #     # Return a random element from a list: 随机选一个MBS
    #     #
    #     #     r = self.sce['r'+BS_type] * random()  # MBS的半径 500米
    #     #     theta = uniform(-pi, pi)  # theta
    #     #     Loc_agent[0] = LocM[0] + r * np.cos(theta)  # 随机选取位置
    #     #     Loc_agent[1] = LocM[1] + r * np.sin(theta)  #
    #     #     return Loc_agent

    def Get_Location(self):
        return self.location

    # def sample(self):
    #     '''Only sample from the remaining valid spaces
    #     '''
    #     if len(self.valid_spaces) == 0:
    #         print("Space is empty")
    #         return None
    #     np_random, _ = seeding.np_random()
    #     randint = np_random.randint(len(self.valid_spaces))
    #     return self.valid_spaces[randint]


# gym.Env
class Environment(gym.Env):
    def __init__(self, sce: dict):  # Initialize the scenario we simulate

        self.seed = 777
        self.sce = sce
        self.BS_num = self.sce.nMBS + self.sce.nPBS + self.sce.nFBS
        self.nBS = self.BS_num
        self.nRB = self.sce.nRBs
        self.nUE = self.sce.nUEs
        self.BSs = self.BS_Init()
        self.UEs = self.UE_Init()
        self.user_candidate_assignment()
        self.distance_matrix = self.get_distance_matrix()
        self.burst_prob = sce.get("burst_prob", None)  # 用户数据请求的概率
        self.isBurstScenario = True
        # for eval callback. In eval_mode is true, the reward model is objective value instead of other reward model such as the difference between adjacent time slot rewards
        self.eval_mode = False
        if self.isBurstScenario and self.burst_prob is not None:
            # assert self.burst_prob is not None
            print("=" * 10, 'Note: this is the burst scenario', f"self.burst_prob: {self.burst_prob}", "=" * 10)
        else:
            self.isBurstScenario = False
        if sce.prt:
            self.showmap()

        self.init_ue_serving_BSs()  # 新增初始化调用

    def init_ue_serving_BSs(self):
        """基于现有基站UE_set初始化每个UE的关联基站列表"""
        # 创建BS_ID到BS对象的快速映射
        bs_id_map = {bs.id: bs for bs in self.BSs}

        for ue in self.UEs:
            # 为每个UE初始化服务基站列表
            if not hasattr(ue, 'serving_BSs'):
                ue.serving_BSs = set()

            # 遍历所有基站寻找包含该UE的BS
            for bs in self.BSs:
                if ue.id in bs.UE_set:
                    ue.serving_BSs.add(bs.id)

            # 若UE没有被任何基站覆盖，自动关联最近基站（容错逻辑）
            if len(ue.serving_BSs) == 0:
                print('environment.py: UE没有被任何基站覆盖，自动关联最近基站（容错逻辑）')
                closest_bs = min(self.BSs,
                                 key=lambda b: np.linalg.norm(
                                     np.array(b.BS_Loc) - np.array(ue.location)))
                ue.serving_BSs.add(closest_bs.id)
                closest_bs.UE_set.append(ue.id)  # 更新BS的UE_set
    def get_distance_matrix(self):
        distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
        print('distance matrix is calculated!')
        return distance_matrix

    def get_n0(self):
        # return the environmental noise # note: not dB
        return 10 ** (self.sce.N0 / 10) * self.sce.BW
    def get_n0_dB(self):
        # return the environmental noise # note: not dB
        return self.sce.N0 * self.sce.BW
    def get_estimated_H(self, H: np.ndarray, error_percentage: float):
        """

        :param H: input H should be real number unit, not dB unit.
        :param error_percentage:
        :return:
        """
        # 设置估计误差比例
        # error_percentage = 0.05  # 5%

        # 生成噪声矩阵，噪声的标准差是原始矩阵值的 5%（相对误差） 加性噪声
        noise_matrix = np.random.normal(loc=0, scale=error_percentage * np.abs(H), size=H.shape)
        estimated_H = H + noise_matrix
        estimated_H = np.abs(estimated_H)
        estimated_H = np.maximum(estimated_H, 1e-10)
        # 生成乘性噪声（1 ± error_percentage）
        # noise_factor = 1 + np.random.normal(loc=0, scale=error_percentage, size=H.shape)
        # estimated_H = H * noise_factor
        # estimated_H = np.maximum(estimated_H, 1e-15)
        return estimated_H

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.distance_matrix = self.get_distance_matrix()

    def showmap(self, path: str = None):
        """
        Visualize the base and user location
        :param path:  path to save the figure
        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        # if path is None:
        #     path = "../my_env_plot.png"
        for b in self.BSs:
            print(f'BS id: {b.id}, UE_Set: {b.UE_set}\n')
        base_positions = []
        user_positions = []
        user_original_postions = []
        for b in self.BSs:
            base_positions.append(b.BS_Loc)
        for u in self.UEs:
            try:
                user_original_postions.append(u.original_location)
                user_positions.append(u.location)
            except Exception as e:
                user_original_postions.append(u.location)
                user_positions.append(u.location)
        # 设定基站的位置
        # base_positions = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]])
        base_positions = np.array(base_positions)

        # 基站的信号覆盖半径
        # def generate_base_positions(n):
        #     if n < 1 or (np.sqrt(8 * n - 7) - 1) / 3 % 1 != 0:
        #         raise ValueError("基站的数量必须是一个完全六边形数，如1、7、19、37")
        #     # 计算m的值
        #     m = int((np.sqrt(8 * n - 7) + 1) // 3)
        #     # 初始化基站位置列表
        #     base_positions = []
        #     for i in range(-m + 1, m):
        #         for j in range(max(-m + 1, -i - m + 1), min(m, -i + m)):
        #             x = i + j / 2
        #             y = j * np.sqrt(10) / 2
        #             base_positions.append([x, y])
        #     return np.array(base_positions)

        # base_positions=generate_base_positions(7)
        # radius = 0.6

        # 每个基站的用户数量
        # num_users_per_base = 10

        # 定义矩形区域的边界
        # x_min, x_max = 0, 1000
        # y_min, y_max = 0, 1000

        # 定义要生成的点的数量
        # num_points = 50

        # 生成均匀分布的随机 x 和 y 坐标
        # x = np.random.uniform(x_min, x_max, num_points)
        # y = np.random.uniform(y_min, y_max, num_points)

        # 用户的位置
        # user_positions = []
        #
        # for base_position in base_positions:
        #     for _ in range(num_users_per_base):
        #         # 在基站的信号覆盖范围内随机选择一个角度和半径
        #         angle = 2 * np.pi * np.random.rand()
        #         r =  100* np.sqrt(np.random.rand())
        #         # 计算用户的位置
        #         user_position = base_position + np.array([r * np.cos(angle), r * np.sin(angle)])
        #         user_positions.append(user_position)
        #

        user_positions = np.array(user_positions)
        user_original_postions = np.array(user_original_postions)
        # 画出基站和用户的位置
        plt.scatter(user_positions[:, 0], user_positions[:, 1], label='user', color='blue')
        # plt.scatter(x, y, label='user')

        plt.scatter(base_positions[:, 0], base_positions[:, 1], label='basestation', color='red')
        # 获取当前的Axes对象
        ax = plt.gca()

        # 在每个点上绘制圆圈
        for idx, (xi, yi), in enumerate(zip(base_positions[:, 0], base_positions[:, 1])):
            circle = Circle((xi, yi), self.BSs[idx].BS_Radius, color='darkred', fill=False, linewidth=1.5, alpha=0.6)
            ax.add_patch(circle)
        print('1')
        # todo radius = 50 ?
        # for idx, (xi, yi), in enumerate(zip(user_original_postions[:, 0], user_original_postions[:, 1])):
        #     circle = Circle((xi, yi), 50, color='darkred', fill=False, linewidth=1, alpha=0.6)
        #     ax.add_patch(circle)
        # 设置坐标轴比例相同，以保持圆形
        ax.set_aspect('equal', adjustable='datalim')
        plt.legend()
        if path is not None:
            plt.savefig(path)
            print('Map figure saved to ' + path)
        plt.show()

    def user_candidate_assignment(self) -> None:
        # add UE in every BS's UE candidate set, where UE added in the
        # set is in the signal coverage zone of the BS.
        b: BS
        u: UE
        for b in self.BSs:
            for u in self.UEs:
                if sqrt((u.location[0] - b.BS_Loc[0]) ** 2 + (u.location[1] - b.BS_Loc[1]) ** 2) <= b.BS_Radius:
                    b.UE_set.append(u.id)

        return

    def UE_Init(self) -> list:
        ue_index = 0
        UEs = []

        # 首先，在矩形区域的中心部分散布一些用户
        x_min, x_max = self.sce.x_max * 0.25, self.sce.x_max * 0.75
        y_min, y_max = self.sce.y_max * 0.25, self.sce.y_max * 0.75
        # 为了体现多智能体协调的能力 在中间散步一些用户
        num_points = int(self.sce.nUEs / 4)
        num_points = 1
        # 生成均匀分布的随机 x 和 y 坐标
        xs = np.random.uniform(x_min, x_max, num_points)
        ys = np.random.uniform(y_min, y_max, num_points)
        for x, y in zip(xs, ys):
            ue_index += 1
            UEs.append(UE(x, y, index=ue_index))
        import random
        def cycle_list(lst):
            while True:
                for item in lst:
                    yield item

        my_list = self.BSs[:]
        cycler = cycle_list(my_list)
        for _ in range(self.sce.nUEs - num_points):
            b = next(cycler)
            LocM = b.BS_Loc
            r = b.BS_Radius * random.uniform(0.5, 1)
            theta = uniform(-pi, pi)
            ue_index += 1
            UEs.append(UE(x=LocM[0] + r * np.cos(theta), y=LocM[1] + r * np.sin(theta), index=ue_index))
        return UEs

    def random_walk(self):
        """
        用户随机移动，且更新距离矩阵
        """
        if not hasattr(self.UEs[0], 'serving_BSs'):
            self.init_ue_serving_BSs()
        # 预先创建基站字典以便快速查找
        if self.bs_dict is None:
            self.bs_dict = {bs.id: bs for bs in self.BSs}

        bs_dict = self.bs_dict
        for ue in self.UEs:
            # 如果用户没有关联的基站，跳过（理论上不会发生）
            if not ue.serving_BSs:
                continue

            max_attempts = 100
            attempt = 0
            new_location = None
            current_x, current_y = ue.location

            while attempt < max_attempts:
                # 生成候选位置，基于当前位置的随机移动（步长0到5米）
                r = 5 * random.uniform(0, 1)
                theta = uniform(-pi, pi)
                delta_x = r * np.cos(theta)
                delta_y = r * np.sin(theta)
                candidate_x = current_x + delta_x
                candidate_y = current_y + delta_y

                # 检查候选位置是否在所有关联基站的覆盖范围内
                valid = True
                for bs_id in ue.serving_BSs:
                    bs = bs_dict.get(bs_id)
                    if not bs:
                        valid = False
                        break
                    dx = candidate_x - bs.BS_Loc[0]
                    dy = candidate_y - bs.BS_Loc[1]
                    distance = sqrt(dx ** 2 + dy ** 2)
                    if distance > bs.BS_Radius:
                        valid = False
                        break

                if valid:
                    new_location = [candidate_x, candidate_y]
                    break  # 找到有效位置，退出循环
                attempt += 1

            # 更新用户位置
            if new_location:
                ue.location = new_location
                # 更新该用户在距离矩阵中的所有基站距离
                ue_index = self.UEs.index(ue)  # 获取用户索引
                for b_index, bs in enumerate(self.BSs):
                    dx = new_location[0] - bs.BS_Loc[0]
                    dy = new_location[1] - bs.BS_Loc[1]
                    self.distance_matrix[b_index][ue_index] = sqrt(dx ** 2 + dy ** 2)

    def BS_Init(self) -> list:  # Initialize all the base stations
        BaseStations = []  # The vector of base stations
        Loc_MBS, Loc_PBS, Loc_FBS = self.singleBS_Location()

        for i in range(self.sce.nMBS):  # Initialize the MBSs
            BS_index = i
            BS_type = "MBS"
            BS_Loc = Loc_MBS[i]
            BS_Radius = self.sce.rMBS
            Tx_Power_dBm = self.sce.txpowerMBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        for i in range(self.sce.nPBS):
            BS_index = self.sce.nMBS + i
            BS_type = "PBS"
            BS_Loc = Loc_PBS[i]
            BS_Radius = self.sce.rPBS
            Tx_Power_dBm = self.sce.txpowerPBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        for i in range(self.sce.nFBS):
            BS_index = self.sce.nMBS + self.sce.nPBS + i
            BS_type = "FBS"
            BS_Loc = Loc_FBS[i]
            BS_Radius = self.sce.rFBS
            Tx_Power_dBm = self.sce.txpowerFBSdBm
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius, Tx_Power_dBm))

        return BaseStations

    def BS_Location(self):
        Loc_MBS = np.zeros((self.sce.nMBS, 2))  # Initialize the locations of BSs
        Loc_PBS = np.zeros((self.sce.nPBS, 2))
        Loc_FBS = np.zeros((self.sce.nFBS, 2))
        x_max = self.sce.x_max
        y_max = self.sce.y_max
        bsloclist = self.sce.bsloclist
        if bsloclist is not None:
            assert (self.sce.nMBS + self.sce.nPBS + self.sce.nFBS) == len(bsloclist)
            for i in range(self.sce.nPBS):
                Loc_PBS[i, 0] = bsloclist[i][0]
                Loc_PBS[i, 1] = bsloclist[i][1]

            return Loc_MBS, Loc_PBS, Loc_FBS

        for i in range(self.sce.nMBS):
            # 任意两个MBS之间的间距为x_max
            MBS_deviation = x_max * 0.9  # default 900
            # x_max, default 1000
            Loc_MBS[i, 0] = x_max / 2 + x_max * 0.9 * i  # x-coordinate
            Loc_MBS[i, 1] = y_max / 2  # y-coordinate

        for i in range(self.sce.nPBS):
            # 每个MBS 分配 perMBS 个 PBS, 那么必须满足 self.sce.nPBS > perMBS
            perMBS = 4
            Loc_PBS[i, 0] = Loc_MBS[int(i / perMBS), 0] + x_max / 4 * np.cos(pi / 2 * (i % perMBS))
            Loc_PBS[i, 1] = Loc_MBS[int(i / perMBS), 1] + y_max / 4 * np.sin(pi / 2 * (i % perMBS))

        for i in range(self.sce.nFBS):
            LocM = choice(Loc_MBS)
            r = self.sce.rMBS * random()
            theta = uniform(-pi, pi)
            Loc_FBS[i, 0] = LocM[0] + r * np.cos(theta)
            Loc_FBS[i, 1] = LocM[1] + r * np.sin(theta)

        return Loc_MBS, Loc_PBS, Loc_FBS

    def singleBS_Location(self):
        assert self.sce.nPBS == 1
        assert self.sce.nMBS == 0
        assert self.sce.x_max == 1000 and self.sce.y_max == 1000
        Loc_MBS = np.zeros((self.sce.nMBS, 2))  # Initialize the locations of BSs
        Loc_PBS = np.zeros((self.sce.nPBS, 2))
        Loc_FBS = np.zeros((self.sce.nFBS, 2))

        Loc_PBS[0, 0] = 500
        Loc_PBS[0, 1] = 500

        return Loc_MBS, Loc_PBS, Loc_FBS

    # self.fc = 6  # Frequency in GHz
    # self.d_bp = 4 * 25 * 1.5 / self.fc  # Breakpoint distance

    def path_loss_UMa(self, d_2d, isLOS=True):
        # reference from 3GPPTR38.901ChannelModelChapter_V6_200820.pdf
        # For dense urban areas with high-rise buildings, h_E can be set between 15-30 m
        # For suburban areas with smaller houses, lower h_E of 5-15 m is appropriate
        h_e = 15
        h_ut = 1.5
        h_bs = 25
        fc = self.sce.fc  # Ghz
        c = 3 * 10 ** 8
        d_prime_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc * 10 ** 9 / c  # Urban Macrocell (UMa)
        d_3d = np.sqrt(d_2d ** 2 + (h_bs - h_ut) ** 2)  # Convert to m

        if d_2d <= d_prime_bp:
            r1 = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(fc)
        else:
            r1 = 28.0 + 40 * np.log10(d_3d) + 20 * np.log10(fc) - 9 * np.log10(
                d_prime_bp ** 2 + (h_bs - h_ut) ** 2)
        if not isLOS:
            return max(r1, 13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5))
        else:
            return r1

    # d = np.linspace(10, 5000, 100)
    # pathloss_LOS = [path_loss_UMa(dist) for dist in d]
    # pathloss_NLOS = [path_loss_UMa(dist, False) for dist in d]
    #
    # shadow_fade_LOS = get_shadow_fading(True)
    # shadow_fade_NLOS = get_shadow_fading(False)

    # LOS Probability
    def prob_LOS(self, d):
        if d <= 18:
            return 1
        elif d <= 1000:
            return 18 / d + np.exp(-d / 63) * (1 - 18 / d)
        else:
            return 0

    def generate_correlation_matrix(self, d_c):
        """
        计算协方差矩阵 R，仅计算对称矩阵的一半以提高效率。
        :param distance_matrix: 用户之间的距离矩阵
        :param d_c: 相关性衰减参数
        :return: 协方差矩阵 R
        """
        st = time.time()
        num_users = self.nUE
        R = np.zeros((num_users, num_users))
        for i in range(num_users):
            for j in range(i, num_users):  # 仅计算上三角部分（包含对角线）
                if i == j:
                    R[i, j] = 1  # 对角线元素（与自身的相关性）为 1
                else:
                    Loc_diff = np.array(self.UEs[i].Get_Location()) - self.UEs[j].Get_Location()
                    distance_ij = np.sqrt(Loc_diff[0] ** 2 + Loc_diff[1] ** 2)
                    R[i, j] = np.exp(-distance_ij / d_c)
                    R[j, i] = R[i, j]  # 对称赋值

        et = time.time()
        print(f'distance_UE cal time: {et - st:.2f}')

        R_sqrt = np.linalg.cholesky(R)

        return R, R_sqrt

    def test_cal_Receive_Power(self, bs, d):
        # channel model defined here
        p_LOS = self.prob_LOS(d)
        # if np.random.rand() < p_LOS:
        #     loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # else:
        #     loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)
        # loss_dB = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        assert 0 < p_LOS <= 1
        # loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        loss_dB_LOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=True)
        loss_dB_NLOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=False)
        loss_dB = p_LOS * loss_dB_LOS + (1 - p_LOS) * loss_dB_NLOS

        # loss_dB = 128.1 + 37.6 * np.log10(d / 1000) + 9

        Tx_Power_dBm = bs.Transmit_Power_dBm()
        # if bs.BStype == "MBS" or bs.BStype == "PBS":
        #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
        # elif bs.BStype == "FBS":
        #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()

        if d <= bs.BS_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
            Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
        else:
            Rx_power = 0.0
        H_power_dB = loss_dB
        return Rx_power, H_power_dB

    def test_cal_Receive_Power_without_fading(self, bs, d):
        p_LOS = self.prob_LOS(d)
        if np.random.rand() < p_LOS:
            loss = self.path_loss_UMa(d, isLOS=True)
        else:
            loss = self.path_loss_UMa(d, isLOS=False)
        Tx_Power_dBm = bs.Transmit_Power_dBm()
        # if bs.BStype == "MBS" or bs.BStype == "PBS":
        #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
        # elif bs.BStype == "FBS":
        #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()

        if d <= bs.BS_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss  # Received power in dBm
            Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
        else:
            Rx_power = 0.0
        H_power = loss
        return Rx_power, H_power

    # def test_cal_Receive_Power(self, bs, d):
    #     # d matrix
    #     # bs
    #     # los matrix
    #
    #     p_LOS = self.prob_LOS(d)
    #     if np.random.rand() < p_LOS:
    #         loss_dB = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
    #     else:
    #         loss_dB = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)
    #     Tx_Power_dBm = bs.Transmit_Power_dBm()
    #     # if bs.BStype == "MBS" or bs.BStype == "PBS":
    #     #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
    #     # elif bs.BStype == "FBS":
    #     #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()
    #
    #     if d <= bs.BS_Radius:
    #         Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
    #         Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
    #     else:
    #         Rx_power = 0.0
    #     H_power_dB = loss_dB
    #     return Rx_power, H_power_dB
    # 构造协方差矩阵
    def test_cal_Receive_Power_new(self, bs, d):
        # fading修正为 对信道功率有提升也有下降的情况
        # d matrix
        # bs
        # los matrix
        # import numpy as np
        #
        # # 设置多径数和信号样本
        # num_paths = 10
        # signal = np.array([1+1j, 2+2j, 3+3j])
        #
        # # 生成瑞利衰落系数
        # rayleigh_fading = (np.random.normal(size=num_paths) + 1j * np.random.normal(size=num_paths)) / np.sqrt(2)
        #
        # # 假设信道系数为瑞利衰落系数
        # h = rayleigh_fading
        #
        # # 计算经过衰落后的信号
        # faded_signal = h[:len(signal)] * signal
        #
        # # 计算信道功率
        # channel_power = np.sum(np.abs(h)**2)
        # print("Channel Power with Rayleigh Fading:", channel_power)
        # p_LOS = self.prob_LOS(d)
        # if np.random.rand() < p_LOS:
        #     loss_dB = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # else:
        #     loss_dB = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)
        #
        # Tx_Power_dBm = bs.Transmit_Power_dBm()
        # # if bs.BStype == "MBS" or bs.BStype == "PBS":
        # #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
        # # elif bs.BStype == "FBS":
        # #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()
        #
        # if d <= bs.BS_Radius:
        #     Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
        #     Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
        # else:
        #     Rx_power = 0.0
        # H_power = 10 ** (loss_dB / 10)
        # return Rx_power, H_power
        # channel model defined here
        p_LOS = self.prob_LOS(d)
        # if np.random.rand() < p_LOS:
        #     loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # else:
        #     loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)
        # loss_dB = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        assert 0 < p_LOS <= 1
        # loss_dB_LOS = self.path_loss_UMa(d, isLOS=True) + self.get_shadow_fading_dB(isLOS=True)
        # loss_dB_NLOS = self.path_loss_UMa(d, isLOS=False) + self.get_shadow_fading_dB(isLOS=False)

        loss_dB_LOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=True)
        loss_dB_NLOS = 128.1 + 37.6 * np.log10(d / 1000) + self.get_shadow_fading_dB(isLOS=False)
        loss_dB = p_LOS * loss_dB_LOS + (1 - p_LOS) * loss_dB_NLOS

        # loss_dB = 128.1 + 37.6 * np.log10(d / 1000) + 9

        Tx_Power_dBm = bs.Transmit_Power_dBm()
        # if bs.BStype == "MBS" or bs.BStype == "PBS":
        #     loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
        # elif bs.BStype == "FBS":
        #     loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()

        if d <= bs.BS_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss_dB  # Received power in dBm
            Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
        else:
            Rx_power = 0.0
        H_power_dB = loss_dB
        return Rx_power, H_power_dB

    # def cal_Receive_Power(self, bs, d):  # Calculate the received power by transmit power and path loss of a certain BS
    #     # trivial path loss modules
    #     # p_LOS = self.prob_LOS(d)
    #     #
    #     # if np.random.rand() < p_LOS:
    #     #     loss = self.path_loss_LOS(d,isLOS=True)+self.get_shadow_fading_dB(isLOS=True)
    #     # else:
    #     #     loss = self.path_loss_UMa(d,isLOS=False)+self.get_shadow_fading_dB(isLOS=False)
    #
    #     Tx_Power_dBm = bs.Transmit_Power_dBm()
    #
    #     if bs.BStype == "MBS" or bs.BStype == "PBS":
    #         loss = 34 + 40 * np.log10(d) + self.calc_rayleigh_fading_dB()
    #     elif bs.BStype == "FBS":
    #         loss = 37 + 30 * np.log10(d) + self.calc_rayleigh_fading_dB()
    #
    #     if d <= bs.BS_Radius:
    #         Rx_power_dBm = Tx_Power_dBm - loss  # Received power in dBm
    #         Rx_power = 10 ** (Rx_power_dBm / 10)  # Received power in mW
    #     else:
    #         Rx_power = 0.0
    #     H_power = loss
    #     return Rx_power, H_power
    def get_shadow_fading_dB(self, isLOS, bs_idx=None, ue_idx=None, d_c=None):
        """
        :param isLOS: True or False depending on whether the shadow fading is line of sight or not
        :param d_c: correlation factor for shadow fading between users based on the distance.
        :return:
        """
        std_dev = 4 if isLOS else 6
        z_fading_raw = sqrt(1 / 2) * (
                np.random.normal(loc=0.0, scale=std_dev) + 1j * np.random.normal(loc=0.0, scale=std_dev))
        if d_c is not None:
            # use correlated fading
            dc = 10 if self.sce.dc is None else self.sce.dc
            if self.R_sqrt is None:
                self.R, self.R_sqrt = self.generate_correlation_matrix(d_c=dc)  # cal the covariance matrix
            z_fading_raw = self.R_sqrt @ z_fading_raw  # todo use bsidx and ue idx get the correlated fading
            z_fading_raw = abs(z_fading_raw)  # 复数绝对值即范数

        z_fading = abs(z_fading_raw)
        z_fading_dB = 10 * log10(z_fading ** 2)
        return z_fading_dB

    def calc_rayleigh_fading_dB(self, scale=0.5):

        temp_h = sqrt(1 / 2) * (np.random.normal(loc=0.0, scale=scale) + 1j * np.random.normal(loc=0.0, scale=scale))

        h = abs(temp_h)  # 复数绝对值即范数
        h_dB = 10 * log10(h ** 2)
        return h_dB

    def get_new_CSI_dB(self):
        """

        :return:
        """
        channal_power_set = np.zeros((self.nBS, self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power
        return channal_power_set
    # def cal_SINR(self, action, action_i, state, scenario):  # Get reward for the state-action pair
    #     BS = scenario.Get_BaseStations()
    #     L = scenario.BS_Number()  # The total number of BSs
    #     K = self.sce.nChannel  # The total number of channels
    #
    #     BS_selected = action_i // K
    #     Ch_selected = action_i % K  # Translate to the selected BS and channel based on the selected action index
    #     Loc_diff = BS[BS_selected].Get_Location() - self.location
    #     distance = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))  # Calculate the distance between BS and UE
    #     Rx_power = BS[BS_selected].Receive_Power(distance)  # Calculate the received power
    #
    #     if Rx_power == 0.0:
    #         reward = self.sce.negative_cost  # Out of range of the selected BS, thus obtain a negative reward
    #         QoS = 0  # Definitely, QoS cannot be satisfied
    #     else:  # If inside the coverage, then we will calculate the reward value
    #         Interference = 0.0
    #         for i in range(self.opt.nagents):  # Obtain interference on the same channel
    #             BS_select_i = action[i] // K
    #             Ch_select_i = action[i] % K  # The choice of other users
    #             if Ch_select_i == Ch_selected:  # Calculate the interference on the same channel
    #                 Loc_diff_i = BS[BS_select_i].Get_Location() - self.location
    #                 distance_i = np.sqrt((Loc_diff_i[0] ** 2 + Loc_diff_i[1] ** 2))
    #                 Rx_power_i = BS[BS_select_i].Receive_Power(distance_i)
    #                 Interference += Rx_power_i  # Sum all the interference
    #         Interference -= Rx_power  # Remove the received power from interference
    #         Noise = 10 ** ((self.sce.N0) / 10) * self.sce.BW  # Calculate the noise
    #         SINR = Rx_power / (Interference + Noise)  # Calculate the SINR
    #         Rate = self.sce.BW * np.log2(1 + SINR) / (10 ** 6)  # Calculate the rate of UE
    #         reward = Rate
    #
    #     reward = torch.tensor([reward])
    #     return QoS, reward

# if __name__ == '__main__':

# sce, opt = get_config()
# env = Environment(sce, prt=1)
# 可视化环境的基站和用户位置
# base_positions=[]
# user_positions=[]
# for b in env.BSs:
#     base_positions.append(b.BS_Loc)
# for u in env.UEs:
#     user_positions.append(u.location)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设定基站的位置
# # base_positions = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]])
# base_positions=np.array(base_positions)
# # 基站的信号覆盖半径
#
#
#
# def generate_base_positions(n):
#     if n < 1 or (np.sqrt(8 * n - 7) - 1) / 3 % 1 != 0:
#         raise ValueError("基站的数量必须是一个完全六边形数，如1、7、19、37")
#
#     # 计算m的值
#     m = int((np.sqrt(8 * n - 7) + 1) // 3)
#
#     # 初始化基站位置列表
#     base_positions = []
#
#     for i in range(-m + 1, m):
#         for j in range(max(-m + 1, -i - m + 1), min(m, -i + m)):
#             x = i + j / 2
#             y = j * np.sqrt(10) / 2
#             base_positions.append([x, y])
#
#     return np.array(base_positions)
#
#
# # base_positions=generate_base_positions(7)
# # radius = 0.6
#
# # 每个基站的用户数量
# num_users_per_base = 10
#
# # 定义矩形区域的边界
# x_min, x_max = 0, 1000
# y_min, y_max = 0, 1000
#
# # 定义要生成的点的数量
# num_points = 50
#
# # 生成均匀分布的随机 x 和 y 坐标
# x = np.random.uniform(x_min, x_max, num_points)
# y = np.random.uniform(y_min, y_max, num_points)
#
# # 用户的位置
# # user_positions = []
# #
# # for base_position in base_positions:
# #     for _ in range(num_users_per_base):
# #         # 在基站的信号覆盖范围内随机选择一个角度和半径
# #         angle = 2 * np.pi * np.random.rand()
# #         r =  100* np.sqrt(np.random.rand())
# #         # 计算用户的位置
# #         user_position = base_position + np.array([r * np.cos(angle), r * np.sin(angle)])
# #         user_positions.append(user_position)
# #
# user_positions = np.array(user_positions)
#
# # 画出基站和用户的位置
# plt.scatter(user_positions[:, 0], user_positions[:, 1], label='user')
# # plt.scatter(x, y, label='user')
#
# plt.scatter(base_positions[:, 0], base_positions[:, 1], label='base', color='r')
# plt.legend()
# plt.show()
