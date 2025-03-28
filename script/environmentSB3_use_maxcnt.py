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
    def __init__(self, args):
        # only for single BS environment
        # action space:  gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # note: Given the whole channel state information (observation_space), the agent outputs One UE-RBG pair in each step.

        super().__init__(args)
        self._history_channel_information = None
        self._history_action = None

        self.cnt = 0  # this count is designed for determining the maximum output pair of the decision sequence.
        self.maxcnt = 50
        self.dtype = np.float32
        self.distance_matrix = np.zeros((len(self.BSs), len(self.UEs)))

        for b_index, b in enumerate(self.BSs):
            for ue_index, ue in enumerate(self.UEs):
                Loc_diff = b.Get_Location() - ue.Get_Location()
                self.distance_matrix[b_index][ue_index] = np.sqrt((Loc_diff[0] ** 2 + Loc_diff[1] ** 2))

        self.nUE = args.nUEs
        self.nRB = args.nRBs
        # action: one UE*RB pair
        self.action_space = gym.spaces.discrete.Discrete(n=self.nUE * self.nRB, start=0)
        # obs: [Channel state information + action dimension (last decision pair)]
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(self.nUE * self.nRB * 2,),
                                                    dtype=self.dtype)

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cal_sumrate(self, rbg_decision, get_new_CSI=False):
        """
        compute the sum rate of the whole network given the RBG allocation action
        :param action: RBG allocation decision, dimension: |UE|*|RBG|
        :return: total sum-rate (i.e. log(1+SINR) ) of the communication network
        """
        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW  # Calculate the noise

        action = rbg_decision
        # self.history_action[index_action//self.nRB, index_action%self.nRB] = 1
        action = action.reshape(self.BS_num, self.nUE, self.nRB)
        signal_power_set = np.zeros((self.sce.nRBs, self.sce.nUEs))
        if get_new_CSI:
            channal_power_set = np.zeros((self.BS_num, self.sce.nUEs, self.sce.nRBs))
        else:
            channal_power_set = None

        assert self._history_channel_information is not None
        H_dB = self._history_channel_information.reshape((self.sce.nUEs, self.sce.nRBs), )

        for b_index, b in enumerate(self.BSs):
            for global_u_index in range(self.nUE):  # notice that UE_id starts from 1
                for rb_index in range(self.sce.nRBs):
                    a_b_k_u = action[b_index, global_u_index, rb_index]  # todo working right now
                    if get_new_CSI:
                        _, channel_power_dBm = self.test_cal_Receive_Power(b, self.distance_matrix[b_index][global_u_index])
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

    def step(self, action):
        # the action is an integer that is between 0 and # of nRB*nUE indexing which RB and UE should be paired
        self._history_action[action] = 1

        if self.cnt >= self.maxcnt:
            total_rate, channal_power_set = self.cal_sumrate(self._history_action, get_new_CSI=True)
            # update the CSI of environment
            self._history_channel_information = channal_power_set.reshape(-1, )
            self._history_action = np.zeros_like(self._history_channel_information)
            self.cnt = 0
        else:
            total_rate, _ = self.cal_sumrate(self._history_action, get_new_CSI=False)
            # self.history_channel_information don't change
            self.cnt += 1
        reward = total_rate
        new_obs = np.concatenate([self._history_channel_information, self._history_action.reshape(-1, )], axis=-1)
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

        H = channal_power_set.reshape(-1, )
        self._history_channel_information = H

        empty_action = np.zeros_like(H)
        obs = np.concatenate([H, empty_action], axis=-1)
        self._history_action = empty_action
        observation, info = np.array(obs), {}
        self.cnt = 0
        return observation, info
