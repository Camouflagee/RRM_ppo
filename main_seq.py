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

from module.mycallbacks import SeqPPOEvalCallback
from policy.sequence_ppo import SequencePPO
from utils import *
import warnings

warnings.filterwarnings("ignore")
expName = 'BS1UE20'

expNo = 'E9'  # same expNo has same initialized model parameters
_version = 'seqPPO'
episode_length = 50
_load_env = 1
_load_model = 0

log_folder, eval_log_dir = get_TimeLogEvalDir(
    model_name=_version,
    args=os.path.join(expName, expNo),
    info_str=f'')
# load or create environment/model
with open('config/config_environment_setting.yaml', 'r') as file:
    env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
with open('config/config_training_parameters.yaml', 'r') as file:
    tr_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))

if _load_env:
    unwrapped_env = load_env('saved_env/BS1UE20/Env.zip')
    init_env=SequenceDecisionEnvironmentSB3(env_args)
    init_env.__setstate__(unwrapped_env.__getstate__())
    unwrapped_env = init_env
else:
    unwrapped_env = SequenceDecisionEnvironmentSB3(env_args)
    save_model_env(log_folder, _version, '', None, unwrapped_env)

env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)

if _load_model:
    model = SequencePPO.load(
        path=
        'Experiment_result/seqPPO/BS1UE20/E2/date20250207time160611/model_saves/seqPPO_NumSteps_251904.zip',
        env=env,
        **tr_args,
    )
else:
    model = SequencePPO("MlpPolicy", env, verbose=1, device='cpu', **tr_args, )

# ----------------------manually test the model-------------------------
# test_env = TimeLimit(unwrapped_env, max_episode_steps=100)
# action_list = []
# for _ in range(10):
#     obs, _ = test_env.reset()
#     truncated = False
#     rbue_pair_path = []
#     print('action_path: ', end="")
#     while not truncated:
#         action, _ = model.predict(observation=obs, deterministic=False)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         print(action, end=",")
#         rbue_pair_path.append(action)
#         if truncated:
#             # print(env.history_action)
#             print()
#             print(f"reward: {reward:.2f}", )
#             action_list.append({'act': rbue_pair_path,
#                                 'reward': reward}
#                                )
# ----------------------------------------------------------------------------------------------

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
eval_callback = SeqPPOEvalCallback(eval_env=Monitor(copy.deepcopy(env)),
                                   best_model_save_path=eval_log_dir,
                                   eval_freq=episode_length*10,
                                   n_eval_episodes=2, deterministic=False,
                                   render=False, verbose=1,
                                   )
# set logger
logger = configure(log_folder, ["tensorboard", "stdout", "csv"])
model.set_logger(logger)

# train the model
model.learn(total_timesteps=episode_length * 5000, progress_bar=True, log_interval=10,
            callback=eval_callback,
            reset_num_timesteps=False)

# save model
save_model_env(log_folder, _version, '', model, None)
print('training is done')


# todo 模型输出的pair最大长度是否可以让模型自己决定?

#
# print('system will be shut down in 300s')
# system_shutdown(300)
