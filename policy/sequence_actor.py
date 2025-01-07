from stable_baselines3.common.policies import ActorCriticPolicy


class SequenceActor(ActorCriticPolicy):
    def __init__(self):
        super(SequenceActor, self).__init__()
