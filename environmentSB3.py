from typing import Any
import copy
import gymnasium as gym
import numpy as np
import torch as th

from environment import Environment
from utils import min_max_normalize


class EnvironmentSB3(Environment):
    def __init__(self, sce):
        # only for single BS environment
        # action space: gym.spaces.box.Box(low=0, high=1, shape=(self.nUE*self.nRB,), dtype=self.dtype)
        # note: we relax the discrete action into continue action, each BS-UE pair is modeled by a Normal distribution.
        super().__init__(sce)
        self.last_reward = 0
        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        self.nUE = sce.nUEs
        self.nRB = sce.nRBs
        self.action_space = gym.spaces.box.Box(low=0, high=1, shape=(self.nUE * self.nRB,), dtype=self.dtype)

        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB,),
                                                    dtype=self.dtype)
        self.isBurstSenario = False

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cal_sumrate(self, action):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e., log(1+SINR) ) of the communication network
        """
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        action = action.reshape(self.nUE, self.nRB)
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))

        assert self.history_channel_information is not None
        H_dB = self.history_channel_information.reshape((self.sce.nUEs, self.sce.nRBs), )

        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):  # notice that UE_id starts from 1
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = action[global_u_index, rb_index]  # todo working right now
                    _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    signal_power_set[rb_index][global_u_index] += a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[global_u_index, rb_index] / 10))
                    # 注意 H_dB 是 fading - pathloss
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm

        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        # obs = np.mean(channal_power_set, axis=-1, dtype=self.dtype)
        obs = channal_power_set.reshape(-1, )
        reward = total_rate
        self.history_channel_information = obs
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def cal_sumrate_burst(self, action):
        """
        Compute the sum rate of the whole network given the RBG allocation action,
        considering user burst probability.
        """
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise
        action = action.reshape(self.nUE, self.nRB)

        # 用户 burst 状态：1 表示有数据请求，0 表示无数据请求
        user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        burst_mask = user_burst.astype(np.float32).reshape(-1, 1)  # Shape: (nUE, 1)

        # 初始化信号功率和干扰
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))

        assert self.history_channel_information is not None
        H_dB = self.history_channel_information.reshape((self.sce.nUEs, self.sce.nRBs), )

        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):  # notice that UE_id starts from 1
                if user_burst[global_u_index] == 0:  # 如果用户没有数据请求，跳过
                    continue
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = action[global_u_index, rb_index]  # 当前资源分配
                    _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][
                        global_u_index])
                    signal_power_set[rb_index][global_u_index] += a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[global_u_index, rb_index] / 10))
                    channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm

        # 计算干扰和速率
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + (signal_power_set * burst_mask.T) / interference_m)  # 仅考虑有数据请求的用户
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        obs = channal_power_set.reshape(-1, )
        reward = total_rate
        self.history_channel_information = obs
        terminated, truncated, info = False, False, {}

        return obs, reward, terminated, truncated, info

    def step(self, actions):
        return self.cal_sumrate(actions)

    def reset(self, seed=None, options=None):
        action = self.action_space.sample().reshape(self.nUE, self.nRB)
        channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            # s_assignment_bs = bs_wise_user_association_action[b_index].squeeze()
            # a_assignment_bs = bs_wise_RBG_assignment_action[b_index]
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[global_u_index][rb_index] = channel_power
        obs = channal_power_set.reshape(-1, )
        self.history_channel_information = obs
        observation, info = np.array(obs), {}
        self.last_reward = 0
        return observation, info


class SequenceDecisionEnvironmentSB3(Environment):
    def __init__(self, args):
        # only for single BS environment
        # action space:  gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # note: Given the whole channel state information (observation_space), the agent outputs One UE-RBG pair in each step.

        super().__init__(args)
        self.last_total_rate = 0
        self.history_channel_information = None  # unit: dBm
        self.history_action = None
        self.pairs_cnt = 0  # count the # of the total action pairs per episode
        self.maxcnt = None
        self.dtype = np.float32
        self.set_obs_act_space()
        if self.isBurstScenario:
            # 用户 burst 状态：1 表示有数据请求，0 表示无数据请求
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.episode_cnt = 0
        # burst_mask = user_burst.astype(np.float32).reshape(-1, 1)  # Shape: (nUE, 1)
        # self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        #
        # for b_index, b in enumerate(self.BSs):
        #     for ue_index, ue in enumerate(self.UEs):
        #         Loc_diff = b.Get_Location() - ue.Get_Location()
        #         self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        # self.nUE = args.nUEs
        # self.nRB = args.nRBs
        # # action: one UE*RB pair
        # self.action_space = gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # # obs: [Channel state information + action dimension (last decision pair)]
        # self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB * 2,),
        #                                             dtype=self.dtype)
    def encode_obs_MM(self, H_sq, action):
        """
        参考baseline_MM.py中coeff计算的方式
        https://xirgjz6svzu.feishu.cn/docx/NrevdLdLJoxeBTxU52RcqBNNn2N?from=from_copylink
        :param H_sq: channel
        :param action: solution a_{k,u}
        :return:
        """
        #todo check the nUE*nRB or nRB*nUE
        H_sq = H_sq.reshape(self.nUE, self.nRB).transpose()
        action = action.reshape(self.nUE, self.nRB).transpose()
        n0 = self.get_n0()
        a = action
        C = H_sq  # c_ku for all k, u
        # 计算干扰项 (对于每个k,u，计算sum_{u'≠u} a[k,u']*P[k,u']*H_sq[k,u'])
        # 使用广播技巧
        interference = (a * C).sum(axis=1, keepdims=True) - a * C

        # 计算gamma
        denominator = interference + n0
        gamma = np.where(denominator != 0, (a * C) / denominator, 0)

        # 计算A
        A = a * C + interference + n0

        # 计算系数
        # term1 = C / A (with 0 where A is 0)
        term1 = np.where(A != 0, C / A, 0)

        # term2 = sum_{u'≠u} (gamma[k,u'] * C[k,u']) / A[k,u']
        # 对于每个k,u，计算sum_{u'≠u} gamma[k,u']*C[k,u']/A[k,u']
        # 首先计算每个元素的贡献
        contrib = np.where(A != 0, gamma * C / A, 0)
        # 然后对每个k，计算所有u'≠u的和
        term2 = contrib.sum(axis=1, keepdims=True) - contrib

        coeff = term1 - term2
        return coeff

    def set_obs_act_space(self):
        # set obs and action space based on env's info
        # self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        # for b_index, b in enumerate(self.BSs):
        #     for ue_index, ue in enumerate(self.UEs):
        #         Loc_diff = b.Get_Location() - ue.Get_Location()
        #         self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
        self.nUE = self.sce.nUEs
        self.nRB = self.sce.nRBs
        # action: one UE*RB pair
        self.action_space = gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # obs: [Channel state information + action dimension (last decision pair)]
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB * 2,),
                                                    dtype=self.dtype)

    def setup(self):
        self.set_obs_act_space()
        self.set_user_burst()
        self.eval_mode = False
    def set_user_burst(self):
        if self.isBurstScenario:
            # 用户 burst 状态：1 表示有数据请求，0 表示无数据请求
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.setup()

    def cal_sumrate(self, rbg_decision, get_new_CSI=False):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e. log(1+SINR) ) of the communication network
        """
        Noise = self.get_n0()  # Calculate the noise
        action = rbg_decision
        # self.history_action[index_action//self.nRB, index_action%self.nRB] = 1
        action = action.reshape(self.BS_num, self.nUE, self.nRB)
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))

        if get_new_CSI:
            channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        else:
            channal_power_set = None

        assert self.history_channel_information is not None
        H_dB = self.history_channel_information.reshape((self.sce.nUEs, self.sce.nRBs), )
        H = 10 ** (H_dB / 10)
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):  # notice that UE_id starts from 1
                if self.isBurstScenario and self.user_burst[global_u_index] == 0:  # 如果用户没有数据请求，跳过
                    continue
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = action[b_index, global_u_index, rb_index]  # todo working right now
                    if get_new_CSI:
                        _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][
                            global_u_index])
                        channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm
                    # 注意 H_dB 是 fading - pathloss
                    signal_power_set[rb_index][global_u_index] += a_b_k_u * b.Transmit_Power() / (
                            10 ** (H_dB[global_u_index, rb_index] / 10))

        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)

        return total_rate, channal_power_set

    def cal_sumrate_givenH(self, rbg_decision, H, get_new_CSI=False):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e. log(1+SINR) ) of the communication network
        """
        Noise = self.get_n0()  # Calculate the noise

        action = rbg_decision
        # self.history_action[index_action//self.nRB, index_action%self.nRB] = 1
        action = action.reshape(self.BS_num, self.nUE, self.nRB)
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        if get_new_CSI:
            channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        else:
            channal_power_set = None
        # history_channel_information = H
        H_dB = H.reshape((self.sce.nUEs, self.sce.nRBs), )
        H = 10 ** (H_dB / 10)
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):  # notice that UE_id starts from 1
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = action[b_index, global_u_index, rb_index]
                    if get_new_CSI:
                        _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][
                            global_u_index])
                        channal_power_set[b_index][global_u_index][rb_index] = channel_power_dBm
                    # 注意 H_dB 是 fading - pathloss
                    signal_power_set[rb_index][global_u_index] += (a_b_k_u * b.Transmit_Power() /
                                                                   H[global_u_index, rb_index])
        # channel_power
        interference_sum = signal_power_set.sum(axis=1).reshape(-1, 1)
        interference_sum_m = np.tile(interference_sum, self.sce.nUEs)
        interference_m = interference_sum_m - signal_power_set + Noise
        unscale_rate_m = np.log2(1 + signal_power_set / interference_m)
        total_rate = self.sce.BW * np.sum(unscale_rate_m) / (10 ** 6)
        return total_rate, channal_power_set

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1
        # debugg=sum(self.history_action)
        # if self.pairs_cnt >= self.maxcnt:
        #     total_rate, channal_power_set = self.cal_sumrate(self.history_action, get_new_CSI=True)
        #     # update the CSI of environment
        #     self.history_channel_information = channal_power_set.reshape(-1, )
        #     self.history_action = np.zeros_like(self.history_channel_information)
        #     self.pairs_cnt = 0
        # else:
        total_rate, _ = self.cal_sumrate(self.history_action, get_new_CSI=False)
        # self.history_channel_information don't change
        self.pairs_cnt += 1
        # # reward model2: r = obj_t- obj_t-1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate

        # reward model1: r = obj_t
        # reward = total_rate
        if self.eval_mode:
            reward = total_rate
        new_obs = np.concatenate([self.history_channel_information, self.history_action.reshape(-1, )], axis=-1)
        terminated, truncated, info = False, False, {}
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        # todo can we optimize this code ?
        channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[global_u_index][rb_index] = channel_power
        self.last_total_rate = 0
        H = channal_power_set.reshape(-1, )
        self.history_channel_information = H  # dBm
        self.episode_cnt += 1
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        empty_action = np.zeros_like(H)
        obs = np.concatenate([H, empty_action], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {}
        self.pairs_cnt = 0
        return observation, info

    def reset_onlyforbaseline(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power_dB \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[global_u_index][rb_index] = channel_power_dB

        H = channal_power_set.reshape(-1, )
        self.history_channel_information = H

        empty_action = np.zeros_like(H)
        obs = np.concatenate([H, empty_action], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H}
        self.pairs_cnt = 0
        return observation, info


class SequenceDecisionAdaptiveEnvironmentSB3(SequenceDecisionEnvironmentSB3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_channel_information_error = None
        self.error_percent=None
        self.use_sideinfo=None
    def set_obs_act_space(self):
        # set obs and action space based on env's info
        # self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))
        # for b_index, b in enumerate(self.BSs):
        #     for ue_index, ue in enumerate(self.UEs):
        #         Loc_diff = b.Get_Location() - ue.Get_Location()
        #         self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))
        self.nUE = self.sce.nUEs
        self.nRB = self.sce.nRBs
        # action: one UE*RB pair
        self.action_space = gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)  # +1 : add side information
        # obs: [Channel state information + action dimension (last decision pair)]
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB * 2 + 1,),
                                                    dtype=self.dtype)
        self.history_channel_information_error = None

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1

        total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information, get_new_CSI=False)
        total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error, get_new_CSI=False)

        if self.use_sideinfo:
            channel_damage_info=np.array([total_rate-total_rate_error])
        else:
            channel_damage_info=np.array([0])

        # self.history_channel_information don't change
        self.cnt += 1
        # # reward model2: r = obj_t- obj_t-1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate

        # reward model1: r = obj_t
        # reward = total_rate
        if self.eval_mode:
            reward = total_rate


        new_obs = np.concatenate([self.history_channel_information_error_norm, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
        terminated, truncated, info = False, False, {}

        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        # todo can we optimize this code ?
        channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[global_u_index][rb_index] = channel_power
        self.last_total_rate = 0
        H_dB = channal_power_set.reshape(-1, )
        H_uk= 10 ** (H_dB/10)
        H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
        H_error_dB = 10*np.log10(H_error_uk)
        H_error_dB_norm = min_max_normalize(H_error_dB)
        self.history_channel_information_error_norm = H_error_dB_norm
        self.history_channel_information_error = H_error_dB # dBm
        self.history_channel_information = H_dB  # dBm
        self.episode_cnt += 1
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB_norm, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {}
        self.cnt = 0
        return observation, info
    def reset_onlyforbaseline(self, given_obs=None,seed=None, options=None, _error_percent=None):
        if not given_obs:
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            # generate new H
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk= 10 ** (H_dB/10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10*np.log10(H_error_uk)
                                                # 'CSI_error_': self.get_estimated_H(H, self.error_percent),}
        else:
            H_dB=given_obs[1]['CSI']
            if _error_percent:
                H_uk = 10 ** (H_dB / 10)
                H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
                H_error_dB = 10 * np.log10(H_error_uk)
            else:
                H_error_dB = given_obs[1]['CSI_error']
        self.history_channel_information = H_error_dB # dBm
        self.history_channel_information_true = H_dB  # dBm
        self.episode_cnt += 1
        self.last_total_rate = 0
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.cnt = 0
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H_dB,
                                            'CSI_error': H_error_dB,}
        return observation, info

class MMSequenceDecisionAdaptiveEnvironmentSB3(SequenceDecisionEnvironmentSB3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_channel_information_error = None
        self.error_percent = None
        self.use_sideinfo = None
    # def encode_obs_MM(self, H_sq, action):
    #     """
    #     :param H_sq: channel
    #     :param action: solution a_{k,u}
    #     :return:
    #     """
    #     n0 = self.get_n0()
    #     a = action
    #     C = H_sq  # c_ku for all k, u
    #     # 计算干扰项 (对于每个k,u，计算sum_{u'≠u} a[k,u']*P[k,u']*H_sq[k,u'])
    #     # 使用广播技巧
    #     # if len(a.shape)<2 or len(C.shape)<2:
    #     #     a=a.reshape(self.nUE,self.nRB)
    #     #     C=C.reshape(self.nUE,self.nRB)
    #     interference = (a * C).sum(axis=1, keepdims=True) - a * C
    #
    #     # 计算gamma
    #     denominator = interference + n0
    #     gamma = np.where(denominator != 0, (a * C) / denominator, 0)
    #
    #     # 计算A
    #     A = a * C + interference + n0
    #
    #     # 计算系数
    #     # term1 = C / A (with 0 where A is 0)
    #     term1 = np.where(A != 0, C / A, 0)
    #
    #     # term2 = sum_{u'≠u} (gamma[k,u'] * C[k,u']) / A[k,u']
    #     # 对于每个k,u，计算sum_{u'≠u} gamma[k,u']*C[k,u']/A[k,u']
    #     # 首先计算每个元素的贡献
    #     contrib = np.where(A != 0, gamma * C / A, 0)
    #     # 然后对每个k，计算所有u'≠u的和
    #     term2 = contrib.sum(axis=1, keepdims=True) - contrib
    #
    #     coeff = term1 - term2
    #     return coeff

    # def step(self, action):
    #     # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
    #     self.history_action[action] = 1
    #
    #     total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information,
    #                                             get_new_CSI=False)
    #     total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error,
    #                                                   get_new_CSI=False)
    #
    #     if self.use_sideinfo:
    #         channel_damage_info = np.array([total_rate - total_rate_error])
    #     else:
    #         channel_damage_info = np.array([0])
    #
    #     # self.history_channel_information don't change
    #     self.pairs_cnt += 1
    #     # # reward model2: r = obj_t- obj_t-1
    #     # reward = total_rate - self.last_total_rate
    #     # self.last_total_rate = total_rate
    #     # reward model3:
    #     mm_coeff=self.encode_obs_MM(self.history_channel_information_error, self.history_action)
    #     mm_coeff_nor = min_max_normalize(mm_coeff)
    #     reward = mm_coeff_nor[action]
    #
    #     # reward model1: r = obj_t
    #     # reward = total_rate
    #     if self.eval_mode:
    #         reward = total_rate
    #
    #     new_obs = np.concatenate(
    #         [mm_coeff_nor, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
    #     # self.history_channel_information_error
    #     terminated, truncated, info = False, False, {'pairs_cnt': sum(self.history_action)}
    #
    #     return new_obs, reward, terminated, truncated, info

    # def reset(self, seed=None, options=None):
    #     # action = self.action_space.sample().reshape(self.nUE, self.nRB)
    #     # todo can we optimize this code ?
    #     channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
    #     for b_index, b in enumerate(self.BSs):
    #         for global_u_index in range(self.nUE):
    #             for rb_index in range(self.nRB):
    #                 signal_power, channel_power \
    #                     = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
    #                 channal_power_set[global_u_index][rb_index] = channel_power
    #     self.last_total_rate = 0
    #     H_dB = channal_power_set.reshape(-1, )
    #     H_uk = 10 ** (H_dB / 10)
    #     H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
    #     H_error_dB = 10 * np.log10(H_error_uk)
    #     self.history_channel_information_error = H_error_dB  # dBm
    #     self.history_channel_information = H_dB  # dBm
    #     self.episode_cnt += 1
    #     if self.isBurstScenario and self.episode_cnt % 10 == 0:
    #         self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
    #     channel_damage_info = np.array([0])
    #     empty_action = np.zeros_like(H_dB) # np.ones_like(H_dB)*0.5
    #     self.history_action = empty_action
    #
    #     #mm
    #     mm_coeff=self.encode_obs_MM(self.history_channel_information_error, self.history_action)
    #     mm_coeff_nor = mm_coeff/mm_coeff.max()
    #
    #     obs = np.concatenate([mm_coeff_nor, empty_action, channel_damage_info], axis=-1)
    #         # H_error_dB
    #     observation, info = np.array(obs), {}
    #     self.pairs_cnt = 0
    #     return observation, info
    def encode_obs_MM(self, H_sq, action):
        """
        参考baseline_MM.py中coeff计算的方式
        https://xirgjz6svzu.feishu.cn/docx/NrevdLdLJoxeBTxU52RcqBNNn2N?from=from_copylink
        :param H_sq: channel [U,K]
        :param action: solution a_{k,u} [U,K]
        :return:
        """
        #todo check the nUE*nRB or nRB*nUE
        # transpose to be suitable for following codes
        H_sq = H_sq.reshape(self.nUE, self.nRB).transpose()
        action = action.reshape(self.nUE, self.nRB).transpose()
        n0 = self.get_n0()
        a = action
        C = H_sq  # c_ku for all k, u
        # 计算干扰项 (对于每个k,u，计算sum_{u'≠u} a[k,u']*P[k,u']*H_sq[k,u'])
        # 使用广播技巧
        interference = (a * C).sum(axis=1, keepdims=True) - a * C

        # 计算gamma
        denominator = interference + n0
        gamma = np.where(denominator != 0, (a * C) / denominator, 0)

        # 计算A
        A = a * C + interference + n0

        # 计算系数
        # term1 = C / A (with 0 where A is 0)
        term1 = np.where(A != 0, C / A, 0)

        # term2 = sum_{u'≠u} (gamma[k,u'] * C[k,u']) / A[k,u']
        # 对于每个k,u，计算sum_{u'≠u} gamma[k,u']*C[k,u']/A[k,u']
        # 首先计算每个元素的贡献
        contrib = np.where(A != 0, gamma * C / A, 0)
        # 然后对每个k，计算所有u'≠u的和
        term2 = contrib.sum(axis=1, keepdims=True) - contrib

        coeff = term1 - term2
        return coeff
    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1

        total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information, get_new_CSI=False)
        total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error, get_new_CSI=False)

        if self.use_sideinfo:
            channel_damage_info=np.array([total_rate-total_rate_error])
        else:
            channel_damage_info=np.array([0])

        # self.history_channel_information don't change
        self.cnt += 1
        # # reward model2: r = obj_t- obj_t-1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate

        encode_mm_rsrp=self.encode_obs_MM(self.history_channel_information_error_uk, self.history_action).reshape(-1)

        # reward = encode_mm_rsrp[action]
        # reward model1: r = obj_t
        # reward = total_rate
        if self.eval_mode:
            reward = total_rate


        new_obs = np.concatenate([encode_mm_rsrp, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
        terminated, truncated, info = False, False, {}

        return new_obs, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        # todo can we optimize this code ?
        if 1 or self.history_channel_information is None or self.history_channel_information_error is None:
            print('Note: channel info is re-produced')
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk = 10 ** (H_dB / 10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10 * np.log10(H_error_uk)
            H_error_dB_norm = min_max_normalize(H_error_dB)
            self.history_channel_information_error_norm = H_error_dB_norm
            self.history_channel_information_error = H_error_dB  # dBm
            self.history_channel_information_error_uk=1/H_error_uk
            self.history_channel_information = H_dB  # dBm

        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)

        self.last_total_rate = 0
        self.episode_cnt += 1
        channel_damage_info = np.array([0])

        empty_action = np.zeros_like(self.history_channel_information)
        self.history_action = empty_action
        obs = np.concatenate([self.encode_obs_MM(self.history_channel_information_error_uk, self.history_action).reshape(-1), empty_action, channel_damage_info], axis=-1)

        observation, info = np.array(obs), {'CSI': self.history_channel_information,
                                            'CSI_error': self.history_channel_information_error, }
        self.cnt = 0
        return observation, info
    def reset_onlyforbaseline(self, given_obs=None, seed=None, options=None, _error_percent=None):
        if not given_obs:
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            # generate new H
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk = 10 ** (H_dB / 10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10 * np.log10(H_error_uk)
            # 'CSI_error_': self.get_estimated_H(H, self.error_percent),}
        else:
            H_dB = given_obs[1]['CSI']
            if _error_percent:
                H_uk = 10 ** (H_dB / 10)
                H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
                H_error_dB = 10 * np.log10(H_error_uk)
            else:
                H_error_dB = given_obs[1]['CSI_error']
        self.history_channel_information = H_error_dB  # dBm
        self.history_channel_information_true = H_dB  # dBm
        self.episode_cnt += 1
        self.last_total_rate = 0
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.cnt = 0
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H_dB,
                                            'CSI_error': H_error_dB, }
        return observation, info

class ReverseSequenceDecisionAdaptiveEnvironmentSB3(SequenceDecisionEnvironmentSB3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_channel_information_error = None
        self.error_percent = None
        self.use_sideinfo = None

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1 - self.history_action[action]

        total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information,
                                                get_new_CSI=False)
        total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error,
                                                      get_new_CSI=False)

        if self.use_sideinfo:
            channel_damage_info = np.array([total_rate - total_rate_error])
        else:
            channel_damage_info = np.array([0])

        # self.history_channel_information don't change
        self.cnt += 1
        # # reward model2: r = obj_t- obj_t-1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate

        # reward model1: r = obj_t
        # reward = total_rate
        if self.eval_mode:
            reward = total_rate

        new_obs = np.concatenate(
            [self.history_channel_information_error, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
        terminated, truncated, info = False, False, {}

        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        # todo can we optimize this code ?
        channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):
                for rb_index in range(self.nRB):
                    signal_power, channel_power \
                        = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                    channal_power_set[global_u_index][rb_index] = channel_power
        self.last_total_rate = 0
        H_dB = channal_power_set.reshape(-1, )
        H_uk = 10 ** (H_dB / 10)
        H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
        H_error_dB = 10 * np.log10(H_error_uk)
        self.history_channel_information_error = H_error_dB  # dBm
        self.history_channel_information = H_dB  # dBm
        self.episode_cnt += 1
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {}
        self.cnt = 0
        return observation, info

    def reset_onlyforbaseline(self, given_obs=None, seed=None, options=None, _error_percent=None):
        if not given_obs:
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            # generate new H
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk = 10 ** (H_dB / 10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10 * np.log10(H_error_uk)
            # 'CSI_error_': self.get_estimated_H(H, self.error_percent),}
        else:
            H_dB = given_obs[1]['CSI']
            if _error_percent:
                H_uk = 10 ** (H_dB / 10)
                H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
                H_error_dB = 10 * np.log10(H_error_uk)
            else:
                H_error_dB = given_obs[1]['CSI_error']
        self.history_channel_information = H_error_dB  # dBm
        self.history_channel_information_true = H_dB  # dBm
        self.episode_cnt += 1
        self.last_total_rate = 0
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.cnt = 0
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H_dB,
                                            'CSI_error': H_error_dB, }
        return observation, info

class FixObsSequenceDecisionAdaptiveEnvironmentSB3(SequenceDecisionAdaptiveEnvironmentSB3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_channel_information_error_norm=None
        self.history_channel_information_norm = None

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1

        total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information, get_new_CSI=False)
        total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error, get_new_CSI=False)

        if self.use_sideinfo:
            channel_damage_info=np.array([total_rate-total_rate_error])
        else:
            channel_damage_info=np.array([0])

        # self.history_channel_information don't change
        self.cnt += 1
        # # reward model2: r = obj_t- obj_t-1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate

        encode_mm_rsrp=self.encode_obs_MM(self.history_channel_information_error_norm, self.history_action).reshape(-1)

        # reward = encode_mm_rsrp[action]
        # reward model1: r = obj_t
        # reward = total_rate
        if self.eval_mode:
            reward = total_rate


        new_obs = np.concatenate([encode_mm_rsrp, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
        terminated, truncated, info = False, False, {}

        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # action = self.action_space.sample().reshape(self.nUE, self.nRB)
        # todo can we optimize this code ?
        if self.history_channel_information is None or self.history_channel_information_error is None:
            print('Note: channel info is re-produced')
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk = 10 ** (H_dB / 10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10 * np.log10(H_error_uk)
            H_error_dB_norm = min_max_normalize(H_error_dB)
            self.history_channel_information_error_norm = H_error_dB_norm
            self.history_channel_information_error = H_error_dB  # dBm
            self.history_channel_information = H_dB  # dBm
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)

        self.last_total_rate = 0
        self.episode_cnt += 1
        channel_damage_info = np.array([0])

        empty_action = np.zeros_like(self.history_channel_information)
        self.history_action = empty_action
        obs = np.concatenate([self.encode_obs_MM(self.history_channel_information_error_norm, self.history_action).reshape(-1), empty_action, channel_damage_info], axis=-1)

        observation, info = np.array(obs), {'CSI': self.history_channel_information,
                                            'CSI_error': self.history_channel_information_error, }
        self.cnt = 0
        return observation, info
    def set_obs(self, obs, info):
        self.reset()
        H_error_dB = info['CSI_error']
        H_dB = info['CSI']
        self.history_channel_information_error = H_error_dB # dBm
        self.history_channel_information_error_norm= min_max_normalize(H_error_dB)
        self.history_channel_information = H_dB  # dBm
        self.history_channel_information_norm = min_max_normalize(H_dB)
        empty_action = np.zeros_like(H_dB)
        self.history_action = empty_action

    def reset_onlyforbaseline(self, given_obs=None,seed=None, options=None, _error_percent=None):
        if not given_obs:
            channal_power_set = np.zeros((self.sce.nUEs, self.sce.nRBs))
            # generate new H
            for b_index, b in enumerate(self.BSs):
                for global_u_index in range(self.nUE):
                    for rb_index in range(self.nRB):
                        signal_power, channel_power \
                            = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
                        channal_power_set[global_u_index][rb_index] = channel_power
            H_dB = channal_power_set.reshape(-1, )
            H_uk= 10 ** (H_dB/10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10*np.log10(H_error_uk)
                                                # 'CSI_error_': self.get_estimated_H(H, self.error_percent),}
        else:
            H_dB=given_obs[1]['CSI']
            if _error_percent:
                H_uk = 10 ** (H_dB / 10)
                H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
                H_error_dB = 10 * np.log10(H_error_uk)
            else:
                H_error_dB = given_obs[1]['CSI_error']
        self.history_channel_information = H_error_dB # dBm
        self.history_channel_information_true = H_dB  # dBm
        self.episode_cnt += 1
        self.last_total_rate = 0
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.cnt = 0
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H_dB,
                                            'CSI_error': H_error_dB,}
        return observation, info

class RandomWalkSequenceDecisionAdaptiveEnvironmentSB3(SequenceDecisionAdaptiveEnvironmentSB3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_channel_information_error_norm=None
        self.history_channel_information_norm = None

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self.history_action[action] = 1

        total_rate, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information, get_new_CSI=False)
        total_rate_error, _ = self.cal_sumrate_givenH(self.history_action, self.history_channel_information_error, get_new_CSI=False)

        if self.use_sideinfo:
            channel_damage_info=np.array([total_rate-total_rate_error])
        else:
            channel_damage_info=np.array([0])

        # self.history_channel_information don't change
        self.pairs_cnt += 1
        # reward model 1
        reward = total_rate - self.last_total_rate
        self.last_total_rate = total_rate
        encode_mm_rsrp=self.encode_obs_MM(self.history_channel_information_error_norm, self.history_action).reshape(-1)

        # reward model 2
        # reward = total_rate

        if self.eval_mode:
            reward = total_rate
        new_obs = np.concatenate([encode_mm_rsrp, self.history_action.reshape(-1, ), channel_damage_info], axis=-1)
        terminated, truncated, info = False, False, {}

        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 1. generate new CSI
        # if self.history_channel_information is None or self.history_channel_information_error is None:
        # self.random_walk()
        CSI_dB_UERB = self.get_new_CSI_dB()
        if len(CSI_dB_UERB.shape) > 3:
            assert CSI_dB_UERB.shape[0] == 1
            CSI_dB_UERB = CSI_dB_UERB.squeeze(0) # convert the shape: [BSs=1,UEs,RBs] to shape: [UEs,RBs]
        # 2. pre-process the CSI # e.g., adding noise
        H_dB = CSI_dB_UERB.reshape(-1, )
        H_uk = 10 ** (H_dB / 10)
        H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
        H_error_dB = 10 * np.log10(H_error_uk)
        H_error_dB_norm = min_max_normalize(H_error_dB)
        self.history_channel_information_error_norm = H_error_dB_norm
        self.history_channel_information_error = H_error_dB  # dBm
        self.history_channel_information = H_dB  # dBm
        # 3. burst scenario
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        # 4. adaptive scenario info: indicator of CSI error level
        channel_damage_info = np.array([0])
        # 5. reset the auxiliary varivable
        self.last_total_rate = 0
        self.episode_cnt += 1
        self.pairs_cnt = 0
        # 6. set the final obs
        empty_action = np.zeros_like(self.history_channel_information)
        self.history_action = empty_action
        obs = np.concatenate([self.encode_obs_MM(self.history_channel_information_error_norm, self.history_action).reshape(-1), empty_action, channel_damage_info], axis=-1)
        observation, info = np.array(obs), {'CSI': self.history_channel_information,
                                            'CSI_error': self.history_channel_information_error, }

        return observation, info

    def reset_onlyforbaseline(self, given_obs=None,seed=None, options=None, _error_percent=None):
        """

        :param given_obs: init the environment instance based on given CSI obs
        :param seed:
        :param options:
        :param _error_percent:
        :return: the CSI info for other baseline methods
        """

        if not given_obs:

            CSI_dB_UERB = self.get_new_CSI_dB()
            if len(CSI_dB_UERB.shape) >= 3:
                assert CSI_dB_UERB.shape[0] == 1
                CSI_dB_UERB = CSI_dB_UERB.squeeze(0)  # convert the shape: [BSs=1,UEs,RBs] to shape: [UEs,RBs]
            H_dB = CSI_dB_UERB.reshape(-1, )
            H_uk= 10 ** (H_dB/10)
            H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
            H_error_dB = 10*np.log10(H_error_uk)
                                                # 'CSI_error_': self.get_estimated_H(H, self.error_percent),}
        else:
            H_dB=given_obs[1]['CSI']
            if _error_percent:
                H_uk = 10 ** (H_dB / 10)
                H_error_uk = self.get_estimated_H(H_uk, self.error_percent)
                H_error_dB = 10 * np.log10(H_error_uk)
            else:
                H_error_dB = given_obs[1]['CSI_error']
        self.history_channel_information = H_error_dB # dBm
        self.history_channel_information_true = H_dB  # dBm
        self.episode_cnt += 1
        self.last_total_rate = 0
        if self.isBurstScenario and self.episode_cnt % 10 == 0:
            self.user_burst = np.random.rand(self.nUE) < self.burst_prob  # Shape: (nUE,)
        self.pairs_cnt = 0
        channel_damage_info = np.array([0])
        empty_action = np.zeros_like(H_dB)
        obs = np.concatenate([H_error_dB, empty_action, channel_damage_info], axis=-1)
        self.history_action = empty_action
        observation, info = np.array(obs), {'CSI': H_dB,
                                            'CSI_error': H_error_dB,}
        return observation, info
