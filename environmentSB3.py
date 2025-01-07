from typing import Any
import copy
import gymnasium as gym
import numpy as np
import torch as th

from environment import Environment


class EnvironmentSB3(Environment):
    def __init__(self, sce):
        # only for single BS environment
        # action space: gym.spaces.box.Box(low=0, high=1, shape=(self.nUE*self.nRB,), dtype=self.dtype)
        # note: we relax the discrete action into continue action, each BS-UE pair is modeled by a Normal distribution.
        super().__init__(sce)

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

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cal_sumrate(self, action):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e. log(1+SINR) ) of the communication network
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

        return observation, info


class SequenceDecisionEnvironmentSB3(Environment):
    def __init__(self, sce):
        # only for single BS environment
        # action space:  gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # note: Given the whole channel state information (observation_space), the agent outputs One UE-RBG pair in each step.

        super().__init__(sce)
        self.history_channel_information = None
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        self.nUE = sce.nUEs
        self.nRB = sce.nRBs
        self.action_space = gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)

        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB,),
                                                    dtype=self.dtype)

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cal_sumrate(self, action):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e. log(1+SINR) ) of the communication network
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

        return observation, info
