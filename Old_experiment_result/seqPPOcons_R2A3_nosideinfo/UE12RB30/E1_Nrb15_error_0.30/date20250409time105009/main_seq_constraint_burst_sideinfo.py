# training script of SequencePPO
import numpy as np
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from environmentSB3 import SequenceDecisionAdaptiveEnvironmentSB3

from module.mycallbacks import SeqPPOEvalCallback
from module.sequencepolicy import SequenceActorCriticPolicy
from module.sequence_ppo import SequencePPO
from utils import *
import warnings
import shutil
warnings.filterwarnings("ignore")


def trainer(total_timesteps, _version, envName, expNo, episode_length, env_args, tr_args, load_env_path,
            load_model_path, isBurst, burstprob, error_percent, usesideinfo):
    time_log_folder, time_eval_log_dir = get_TimeLogEvalDir(
        log_root='Experiment_result',
        model_name=_version,
        args=os.path.join(envName, expNo),
        info_str=f'')
    if load_env_path:
        unwrapped_env = load_env(load_env_path)
        if not isinstance(unwrapped_env, SequenceDecisionAdaptiveEnvironmentSB3):
            init_env = SequenceDecisionAdaptiveEnvironmentSB3(env_args)
            init_env.__setstate__(unwrapped_env.__getstate__())
            unwrapped_env = init_env
    else:
        unwrapped_env = SequenceDecisionAdaptiveEnvironmentSB3(env_args)
    if isBurst and burstprob:
        unwrapped_env.isBurstScenario = isBurst
        unwrapped_env.burst_prob = burstprob
        unwrapped_env.set_user_burst()
    # if error_percent:
    unwrapped_env.error_percent = error_percent
    unwrapped_env.use_sideinfo = usesideinfo
    save_model_env(time_log_folder, _version, '', None, unwrapped_env)

    # 保存env及其环境的图
    collect_rollout_steps = 2048 if episode_length * 5 <= 2048 else episode_length * 5
    env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)
    if load_model_path:
        tr_args['policy'] = SequenceActorCriticPolicy
        model = SequencePPO.load(
            path=load_model_path,
            env=env,
            **tr_args,
        )
    else:
        tr_args['policy_kwargs'] = {'const_args': env_args}
        tr_args.n_steps = collect_rollout_steps
        model = SequencePPO(policy=SequenceActorCriticPolicy, env=env, verbose=1, device='cpu', **tr_args, )

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

    if not os.path.exists(time_log_folder):
        os.makedirs(time_log_folder)
    # ---- save training script
    # import shutil
    # src_path = 'Experiment_result/FACMAC7/2BS10UE/script_new_training_2BS10UE.py'
    # dest_path = os.path.join(log_folder, f'script_training.py')
    # shutil.copy(src_path, dest_path)
    # print(f"文件已经成功从 {src_path} 复制到 {dest_path}")

    with open(os.path.join(time_log_folder, 'config_training_parameters.yaml'), 'w') as file:
        yaml.dump(tr_args, file)
    # 保存当前main代码到实验目录中
    current_script_path = os.path.abspath(__file__)
    target_file_path = os.path.join(time_log_folder, os.path.basename(current_script_path))
    shutil.copy(current_script_path, target_file_path)
    # set logger
    logger = configure(time_log_folder, ["tensorboard", "stdout", "csv"])
    model.set_logger(logger)
    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50*2, min_evals=5, verbose=1)
    eval_env = copy.deepcopy(env)
    eval_env.env.eval_mode = True
    eval_callback = SeqPPOEvalCallback(eval_env=Monitor(eval_env),
                                       best_model_save_path=time_eval_log_dir,
                                       eval_freq=collect_rollout_steps // 2,
                                       n_eval_episodes=8, deterministic=False,
                                       render=False, verbose=1,
                                       # callback_after_eval=stop_train_callback
                                       )
    # train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=10,
                callback=eval_callback,
                reset_num_timesteps=False)

    # save model
    save_model_env(time_log_folder, _version, '', model, None)
    print('training is done')

    # todo 模型输出的pair最大长度是否可以让模型自己决定?
    # print('system will be shut down in 300s')
    # system_shutdown(300)


if __name__ == '__main__':
    # expName = 'BS1UE20'
    _version = 'seqPPOcons_R2A3_nosideinfo'
    # load or create environment/model
    with open('config/config_environment_setting.yaml', 'r') as file:
        _env_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
    with open('config/config_training_parameters.yaml', 'r') as file:
        _tr_args = DotDic(yaml.load(file, Loader=yaml.FullLoader))
    isBurst = False
    isAdaptive = True
    use_sideinfo = False

    burstprob = 0.8
    for idx, (nUE, nRB, Nrb) in enumerate(zip([5, 10, 12, 15], [10, 20, 30, 40], [5, 10, 15, 20])):
        if idx in [0, 1, 3]:
            continue
        _error_percent_list = np.arange(30, 65, 5)/100
        for _error_percent in _error_percent_list:  # 0.01,0.05,0.1,0.15 #0.05, 0.1, 0.
            print(f'UE{nUE}RB{nRB} training - error_percent: {_error_percent:.2f}')
            _episode_length = nUE * Nrb
            _env_args.Nrb = Nrb
            _envName = f'UE{nUE}RB{nRB}'
            _expNo = f'E1_Nrb{_env_args.Nrb}_error_{_error_percent:.2f}'  # same expNo has same initialized model parameters
            _env_args.nUEs = nUE
            _env_args.nRBs = nRB
            _total_timesteps = 400000
            _load_env_path = f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip'
            _load_model_path = None
            trainer(_total_timesteps, _version, _envName, _expNo, _episode_length, _env_args, _tr_args, _load_env_path,
                    _load_model_path, isBurst, burstprob, _error_percent, use_sideinfo)
            print(f'UE{nUE}RB{nRB} training is done')

# 问题1:
# UE少RB多的时候, 在episode_length太长时, 严重影响模型决策
# 1. episode_length长时, 会导致mask掉大部分动作, 导致模型决策出问题.
# info:
# if we use reward model 2, the training step is shorter than those of reward model 1
# reward model 2 = obj_t - obj_(t-1)
# reward model 1 = obj_t
