import copy, os, sys
import yaml
from stable_baselines3.common.logger import configure
from module.marlcustomeEnv import TerminatedState
from module.rl_utils import get_log_eval_dir, learning_rate_schedule, save_model_env, load_env, \
    DotDic
from module.facmac7 import FACMACbase
import warnings
from module.callback import MyEvalCallback
# check list
# env create new or load exist one
# log folder
# MARLstep test_receive_power_without_fading
os.chdir("D:\\PythonProject\MARL-UARL") # desktop
# os.chdir("E:\\pycharmProject\MARL-UARL") # laptop

debug = False

if not debug:
    warnings.filterwarnings("ignore", category=UserWarning, module='gym')
# sce = get_config()
# env_user_association = load_env(
#     "Experiment_result/FACMAC/args1/date20240606time175309/model_saves/env.zip")
# env = MarlCustomEnv4(env_user_association.sce,
#                      env_user_association
#                      )
env = load_env('envs/2BS10UE/2BS10UE.zip')

with open('Experiment_result/FACMAC7/2BS10UE/config_hyperparameter4.yaml', 'r') as file:
    args = DotDic(yaml.load(file, Loader=yaml.FullLoader))

episode_length = args.episode_length
print(args)
testmode=True
expName='2BS10UE'
expNo = '2'
total_timesteps = episode_length * 10 * 20  # 3min/episode
load_model = 1
use_eval_callback = 1
iter_train = 0
pretrain_critic = 0
normal_training = 1
use_action_projection = True
if load_model:
    model_path = 'D:\PythonProject\MARL-UARL\Experiment_result\FACMAC7\\2BS10UE\\2\date20241222time135709\model_saves\eval_best_model\\best_model.zip'
    model = FACMACbase.load(model_path,
                         # the following kwargs is inited by the self.__dict__.update()
                         args=args,
                         train_freq=(args.train_freq, args.train_freq_category),
                         first_training=args.first_training,
                         policy_delay_ua_actor=args.policy_delay_ua_actor,
                         policy_delay_rbg_actor=args.policy_delay_rbg_actor,
                         time_window=None,
                         learning_rate=learning_rate_schedule,
                         learning_starts=args.learning_starts,  # warm-up phase
                         buffer_size=args.replay_buffer_size,
                         gamma=args.discounted_rate,
                         gradient_steps=args.gradient_steps,
                         train_individual_q_index=args.train_individual_q_index,
                         use_equal_mixer=args.use_equal_mixer,  # if 1 use VDN mixer else use qmix
                         if_tr_mixer=args.if_tr_mixer,
                         if_tr_actor=args.if_tr_actor,
                         _stats_window_size=args.stats_window_size,  # ep_info_buffer length
                         target_policy_noise=args.target_action_noise_std,
                         noise_type='normal',
                         action_noise_std_start=0.1,
                         action_noise_std_end=args.action_noise_std,
                         tau=args.tau,
                         verbose=1,
                         )
    model.env.envs[0].env.max_steps = episode_length
    model.env.envs[0].env.use_action_projection = use_action_projection
    env.use_action_projection = use_action_projection
else:
    env.use_action_projection = use_action_projection
    model = FACMACbase(policy="MlpPolicy",
                    env=TerminatedState(env, max_steps=args.episode_length),
                    hyperparameter_dict=args,
                    first_training=False,
                    train_freq=(args.train_freq, args.train_freq_category),
                    policy_delay_ua_actor=args.policy_delay_ua_actor,
                    policy_delay_rbg_actor=args.policy_delay_rbg_actor,
                    time_window=args.time_window,
                    learning_rate=learning_rate_schedule,
                    learning_starts=args.learning_starts,  # warm-up phase
                    buffer_size=args.replay_buffer_size,
                    gamma=args.discounted_rate,
                    batch_size=args.batch_size,
                    gradient_steps=args.gradient_steps,
                    train_individual_q_index=args.train_individual_q_index,
                    use_equal_mixer=args.use_equal_mixer,  # like VDN mixer
                    target_policy_noise=args.target_action_noise_std,
                    if_tr_mixer=args.if_tr_mixer,
                    if_tr_actor=args.if_tr_actor,
                    stats_window_size=args.stats_window_size,  # ep_info_buffer length
                    noise_type='normal',
                    action_noise_std_start=0.1,
                    action_noise_std_end=args.action_noise_std,
                    verbose=1,
                    tau=args.tau,
                    # action_noise=action_noise
                    )
try:
    if normal_training:
        _version="FACMAC7"
        eval_env = copy.deepcopy(env)
        log_folder, eval_log_dir = get_log_eval_dir(
            model_name=_version,
            args=os.path.join(expName, expNo),
            info_str=f'')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        import shutil

        src_path = 'Experiment_result/FACMAC7/2BS10UE/script_new_training_2BS10UE.py'
        dest_path = os.path.join(log_folder, f'script_training.py')
        shutil.copy(src_path, dest_path)
        print(f"文件已经成功从 {src_path} 复制到 {dest_path}")
        with open(os.path.join(log_folder, 'parameters_training.yaml'), 'w') as file:
            yaml.dump(args, file)
        eval_callback = MyEvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                       eval_freq=episode_length * 1,
                                       n_eval_episodes=1, deterministic=True,
                                       render=False, verbose=1, )  # log_path=eval_log_dir,
        model.set_noise(action_std=args.action_noise_std, noise_type='normal')
        logger = configure(log_folder, ["tensorboard", "stdout", "csv"])
        model.set_logger(logger)
        if not load_model:
            save_model_env(log_folder, _version, '', model, None)  # save init point

        model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1,
                    callback=eval_callback,
                    reset_num_timesteps=False)
        save_model_env(log_folder, _version, '', model, None)
# 'Experiment_result/FACMAC3/6/6.1/date20240730time134103/model_saves/eval_best_model/best_model.zip' take a look

except Exception as e:
    # save_model_env(log_folder, _version, '', model, None)
    raise e
finally:
    # save_model_env(log_folder, 'FACMAC3', '', model, None)
    # system_shutdown()
    pass
# todo 用户位置可以改变 目的:使学习到策略具有泛化性,在屏蔽部分用户时其性能不会改变过大.
    #思考点: 如果用户走出了基站的信号范围内 那么基站的动作空间会发生改变 该如何解决?
    # 用户位置改变 完成一半 实现在 environment.random_walk中
        # 待思考的问题 1.用户停留在两个基站信号范围交界处 该如何确定用户随机游走新的位置且保证他始终在两个基站交界处中  答: 当某个用户走出了基站范围 在计算其信道信息时直接置为0 这契合于我们的目的.
        # 2. 只处在单个基站范围内的用户 如何 随机游走不会进入其他基站的范围内?

# todo 基站关联用户后的可视化决策观察
   # 可视化部分完成一半 实现在 environment.configuration_visualization()中


# todo 考虑MAPPO with QMIX?