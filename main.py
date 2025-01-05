import gymnasium as gym
import yaml
import os
from datetime import datetime

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

def get_log_eval_dir(log_root: str = 'Experiment_result', model_name: str = 'FACMAC', args: str = 'args1',
                     time_str: str = None, info_str: str = ''):
    """
    :param info_str:
    :param log_root:
    :param model_name:
    :param args:
    :param time_str:
    :return: log_folder, eval_log_dir
    """

    if time_str is None:
        now = datetime.now()
        time_str = now.strftime("date%Y%m%dtime%H%M%S")
    # f"./{log_root}/{model_name}/{args}/{time_str}" + info_str
    log_folder = os.path.join(log_root,model_name,args,time_str+info_str)
    eval_log_dir = os.path.join(log_root,model_name,args,time_str+info_str,'model_saves','eval_best_model')
            # f'./{log_root}/{model_name}/{args}/{time_str}' + info_str + '/model_saves/eval_best_model/'

    return log_folder, eval_log_dir


# load or create environment
# 1. load configuration

with open('config_environment_setting.yaml', 'r') as file:
    cfg = DotDic(yaml.load(file, Loader=yaml.FullLoader))

env = EnvironmentSB3(cfg)

# load or create model

# train model
model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=10_0000)
model.save('log/model/1')

log_folder, eval_log_dir = get_log_eval_dir(
    model_name=_version,
    args=os.path.join(expName, expNo),
    info_str=f'')

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()