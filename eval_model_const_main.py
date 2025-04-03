import numpy as np
from gymnasium.wrappers import TimeLimit
from policy.sequence_ppo import SequencePPO
from utils import load_env
# burst 场景测试
# 'Experiment_result/seqPPOcons_burst/UE5RB10/E1_Nrb3/date20250320time160404/model_saves/eval_best_model/best_model.zip',
# 'Experiment_result/seqPPOcons_burst/UE10RB20/E1_Nrb3/date20250320time164152/model_saves/eval_best_model/best_model.zip',
# 'Experiment_result/seqPPOcons_burst/UE12RB30/E1_Nrb3/date20250321time110358/model_saves/eval_best_model/best_model.zip',
# # 'Experiment_result/seqPPOcons/UE15RB40/E1_Nrb3/date20250312time215412/model_saves/eval_best_model/best_model.zip'
# unwrapped_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')

#===================================eval for no adaptive environment ===========================================================================================
#     # 'Experiment_result/seqPPOcons_R2A/UE5RB10/E1_Nrb3/date20250325time171636/model_saves/eval_best_model/best_model.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE10RB20/E1_Nrb3/date20250325time190748/model_saves/eval_best_model/best_model.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE12RB30/E1_Nrb3/date20250325time192726/model_saves/eval_best_model/best_model.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE15RB40/E1_Nrb3/date20250325time195327/model_saves/eval_best_model/best_model.zip',
#
#     # 'Experiment_result/seqPPOcons_R2A/UE5RB10/E1_Nrb3/date20250325time171636/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE10RB20/E1_Nrb3/date20250325time190748/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE12RB30/E1_Nrb3/date20250325time192726/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
#     # 'Experiment_result/seqPPOcons_R2A/UE15RB40/E1_Nrb3/date20250325time195327/model_saves/eval_best_model/best_model.zip',
#
#     'Experiment_result/seqPPOcons_R2A2/UE12RB30/E1_Nrb15_error_0/date20250327time235039/model_saves/seqPPOcons_R2A2_NumSteps_401408_.zip',
#     'Experiment_result/seqPPOcons_R2A2/UE12RB30/E1_Nrb15_error_0.05/date20250327time223526/model_saves/seqPPOcons_R2A2_NumSteps_401408_.zip',
#     'Experiment_result/seqPPOcons_R2A2/UE12RB30/E1_Nrb15_error_0.1/date20250327time225614/model_saves/seqPPOcons_R2A2_NumSteps_401408_.zip',
#     'Experiment_result/seqPPOcons_R2A2/UE12RB30/E1_Nrb15_error_0.2/date20250327time231723/model_saves/seqPPOcons_R2A2_NumSteps_401408_.zip',
#
# ]
# model_path_list=['',
#                  '',
# 'Experiment_result/seqPPOcons_fullbuffer/UE12RB30/E1_Nrb15/date20250403time095816/model_saves/eval_best_model/best_model.zip',
#
# ]
# for idx, (nUE, nRB, episode_length,model_path) in enumerate(zip([5, 10, 12, 15][:len(model_path_list)], [10, 20, 30, 40][:len(model_path_list)],[12,21,27,40][:len(model_path_list)],model_path_list)):# 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
#     if idx!=2:
#         continue
#     np.random.seed(0)
#     res=[]
#     num_pair=[]
#     test_num=50
#     unwrapped_env = load_env(f'Experiment_result/seqPPOcons/UE{nUE}RB{nRB}/ENV/env.zip')
#     model = SequencePPO.load(model_path)
#     action_list = []
#     unwrapped_env.eval_mode = True
#     test_env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)
#     for loop in range(test_num):
#         obs, _ = test_env.reset()
#         truncated = False
#         rbue_pair_path = []
#         # print('action_path: ', end="")
#         while not truncated:
#             action, _ = model.predict(observation=obs, deterministic=False)
#             obs, reward, terminated, truncated, info = test_env.step(action)
#             # print(action, end=",")
#             rbue_pair_path.append(action)
#             if truncated:
#                 # print(env.history_action)
#                 res.append(reward)
#                 num_pair.append(sum(obs[nUE*nRB:]))
#                 # print(f"reward: {reward:.2f}", )
#                 # action_list.append({'act': rbue_pair_path,
#                 #                     'reward': reward})
#     print("\n","="*20,f"场景: UE{nUE}RB{nRB}, {test_num}次实验平均后结果","="*20)
#     print("最终目标值：", np.mean(res))
#     print("UE/RB配对数量:", np.mean(num_pair))
#     print("="*20,"done:","="*20)

# ===================================================================================================================================================================================================
# eval for adaptive scenario

model_path_list=[
'Experiment_result/seqPPOcons_R2A2/UE12RB30/E1_Nrb15_error_0.6/date20250403time210037/model_saves/eval_best_model/best_model.zip',
]
# np.random.seed(0)
for idx, (model_path,error_rate) in enumerate(zip(model_path_list,[0.6])):
    Nrb=15
    nUE=12
    nRB=30
    res=[]
    num_pair=[]
    test_num=50
    unwrapped_env = load_env(f'Experiment_result/seqPPOcons_R2A2/UE12RB30/ENV/env.zip')
    model = SequencePPO.load(model_path)
    action_list = []
    unwrapped_env.error_percent=error_rate
    unwrapped_env.eval_mode = True
    test_env = TimeLimit(unwrapped_env, max_episode_steps=nUE*Nrb)
    for loop in range(test_num):
        obs, _ = test_env.reset()
        truncated = False
        rbue_pair_path = []
        # print('action_path: ', end="")
        while not truncated:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, reward, terminated, truncated, info = test_env.step(action)
            # print(action, end=",")
            rbue_pair_path.append(action)
            if truncated:
                # print(env.history_action)
                res.append(reward)
                num_pair.append(sum(obs[nUE*nRB:]))
                # print(f"reward: {reward:.2f}", )
                # action_list.append({'act': rbue_pair_path,
                #                     'reward': reward})
    print("\n","="*20,f"场景: UE{nUE}RB{nRB}_error_{error_rate}, {test_num}次实验平均后结果","="*20)
    print("最终目标值：", np.mean(res))
    print("UE/RB配对数量:", np.mean(num_pair))
    print("="*20,"done:","="*20)