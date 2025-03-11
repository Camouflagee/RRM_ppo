from typing import Tuple

import torch
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from utils import DotDic


class SequenceActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, const_args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(const_args, DotDic):
            self.const_args = DotDic(const_args)
        else:
            self.const_args = const_args

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Consider Constraint Version!
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # Here mean_actions are the flattened logits
        mean_actions = self.action_net(latent_pi)
        masked_mean_actions = self.mask_logits(mean_actions, obs)  # todo
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution = self.action_dist.proba_distribution(action_logits=masked_mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        # nUE,nRB = 10, 20
        # debugg_act=obs.reshape(-1)[nUE*nRB:].round(decimals=2).reshape([nUE,nRB])
        return actions, values, log_prob

    # def mask_logits(self, logits, obs):
    #     """
    #     Note: only for RRM environment # class: SequenceDecisionEnvironmentSB3
    #     mask the actions not to be sampled [determined by argument Nrb] setting logits to -99999
    #     :param logits: logits to be masked
    #     :param obs: generate mask based on obs
    #     obs for sequence decision making is organized as follows: [env_info, action of last time slot]
    #     :return:
    #     """
    #     lst_act_idx = obs.shape[0] - self.const_args.nRBs * self.const_args.nUEs
    #     lst_act = obs[lst_act_idx:]
    #     # mask= #todo
    #     return torch.masked_select(logits, mask) # todo

    def mask_logits(self, logits, obs):
        """
        Mask the actions not to be sampled based on UE-RB constraints.
        :param logits: logits to be masked, shape (UE * RB,)
        :param obs: observation containing environment info and last action info
                    obs = [env_info, last_action], where last_action indicates past allocations
        :return: masked logits
        """
        # Extract relevant dimensions
        n_UEs = self.const_args['nUEs']  # Number of UEs
        n_RBs = self.const_args['nRBs']  # Number of RBs
        obs = obs.reshape(-1)
        # Extract the part of obs that contains the last action information
        lst_act_idx = obs.shape[0] - n_UEs * n_RBs  # obs [1,nUE*nRB]
        lst_act = obs[lst_act_idx:]  # Shape: (UE * RB,)

        # Reshape lst_act to (n_UEs, n_RBs) to represent the UE-RB allocation matrix
        allocation_matrix = lst_act.view(n_UEs, n_RBs)

        # Count the number of RBs already allocated to each UE
        allocated_RBs_per_UE = torch.sum(allocation_matrix, dim=1)  # Shape: (n_UEs,)

        # Create a mask for invalid actions
        mask = torch.zeros_like(allocation_matrix, dtype=torch.bool)  # Shape: (n_UEs, n_RBs)

        # Apply the mask for UEs that have reached their RB allocation limit
        for ue_idx in range(n_UEs):
            if allocated_RBs_per_UE[ue_idx] >= self.const_args.Nrb:  # Check if UE has reached RB limit
                mask[ue_idx, :] = True  # Mask all actions for this UE

        # Flatten the mask to match the logits shape (UE * RB,)
        mask = mask.view(-1)

        # Apply the mask to the logits: set masked logits to a very small value
        masked_logits = logits.clone().reshape(-1)  # Create a copy of logits
        masked_logits[mask] = -99999  # Set masked logits to a very small value
        masked_logits = masked_logits.reshape(1, -1)
        return masked_logits
