# training script of SequencePPO
import copy
import gymnasium as gym
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from environmentSB3 import EnvironmentSB3, SequenceDecisionEnvironmentSB3
from stable_baselines3 import PPO

from policy.sequence_ppo import SequencePPO
from utils import *

expName = 'BS1UE20'
expNo = 'E1'  # same expNo has same initialized model parameters
_version = 'seqPPO'
episode_length = 1200
_load_env = 1
_load_model = 0

log_folder, eval_log_dir = get_log_eval_dir(
    model_name=_version,
    args=os.path.join(expName, expNo),
    info_str=f'')
# load or create environment/model
with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
with open('config/config_training_parameters.yaml', 'r') as file:
    tr_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))

if _load_env:
    env = load_env('saved_env/BS1UE20/SeqEnv.zip')
else:
    env = SequenceDecisionEnvironmentSB3(env_args)
    save_model_env(log_folder, _version, '', None, env)
env = TimeLimit(env, max_episode_steps=episode_length * env.maxcnt)

if _load_model:
    model = PPO.load(
        path=
        'D:\pythonProject\RRM_ppo\Experiment_result\PPO\BS1UE20\E1\date20250106time171834\model_saves\eval_best_model\\best_model.zip',
        env=env,
        **tr_args,
    )
else:
    model = SequencePPO("MlpPolicy", env, verbose=1, device='cpu', **tr_args, )

if not os.path.exists(log_folder):
    os.makedirs(log_folder)
# ---- save training script
# import shutil
# src_path = 'Experiment_result/FACMAC7/2BS10UE/script_new_training_2BS10UE.py'
# dest_path = os.path.join(log_folder, f'script_training.py')
# shutil.copy(src_path, dest_path)
# print(f"文件已经成功从 {src_path} 复制到 {dest_path}")
with open(os.path.join(log_folder, 'config_training_parameters.yaml'), 'w') as file:
    yaml.dump(tr_args, file)
# set eval callback, the action in evaluation mode is determined by the mean of distribution.
eval_callback = EvalCallback(Monitor(copy.deepcopy(env)), best_model_save_path=eval_log_dir,
                             eval_freq=episode_length,
                             n_eval_episodes=1, deterministic=True,
                             render=False, verbose=1, )  # log_path=eval_log_dir,
# set logger
logger = configure(log_folder, ["tensorboard", "stdout", "csv"])
model.set_logger(logger)

# train the model
model.learn(total_timesteps=episode_length*env.maxcnt * 50, progress_bar=True, log_interval=1,
            # callback=eval_callback,
            reset_num_timesteps=False)
# save model
# save_model_env(log_folder, _version, '', model, None)
# print('training is done')
#
# print('system will be shut down in 300s')
# system_shutdown(300)
