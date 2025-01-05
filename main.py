import gymnasium as gym
import yaml


from environmentSB3 import EnvironmentSB3
from stable_baselines3 import PPO

class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
# load or create environment
# 1. load configuration

with open('config_environment_setting.yaml', 'r') as file:
    cfg = DotDic(yaml.load(file, Loader=yaml.FullLoader))

env = EnvironmentSB3(cfg)

# load or create model

# train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()