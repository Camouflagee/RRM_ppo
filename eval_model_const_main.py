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
model_path_list=[
    # 'Experiment_result/seqPPOcons_R2A/UE5RB10/E1_Nrb3/date20250325time171636/model_saves/eval_best_model/best_model.zip',
    # 'Experiment_result/seqPPOcons_R2A/UE10RB20/E1_Nrb3/date20250325time190748/model_saves/eval_best_model/best_model.zip',
    # 'Experiment_result/seqPPOcons_R2A/UE12RB30/E1_Nrb3/date20250325time192726/model_saves/eval_best_model/best_model.zip',
    # 'Experiment_result/seqPPOcons_R2A/UE15RB40/E1_Nrb3/date20250325time195327/model_saves/eval_best_model/best_model.zip',
    'Experiment_result/seqPPOcons_R2A/UE5RB10/E1_Nrb3/date20250325time171636/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
    'Experiment_result/seqPPOcons_R2A/UE10RB20/E1_Nrb3/date20250325time190748/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
    'Experiment_result/seqPPOcons_R2A/UE12RB30/E1_Nrb3/date20250325time192726/model_saves/seqPPOcons_R2A_NumSteps_800768_.zip',
    'Experiment_result/seqPPOcons_R2A/UE15RB40/E1_Nrb3/date20250325time195327/model_saves/eval_best_model/best_model.zip',
]
for idx, (nUE, nRB, episode_length,model_path) in enumerate(zip([5, 10, 12, 15][:len(model_path_list)], [10, 20, 30, 40][:len(model_path_list)],[12,21,27,40][:len(model_path_list)],model_path_list)):# 12,30,27; 10,20,21; 5,10,12; UE,RB,episode_length
    np.random.seed(0)
    res=[]
    num_pair=[]
    test_num=50
    unwrapped_env = load_env(f'Experiment_result/seqPPOcons_BR2A/UE{nUE}RB{nRB}/ENV/env.zip')
    model = SequencePPO.load(model_path)
    action_list = []
    unwrapped_env.eval_mode = True
    test_env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)
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
    print("\n","="*20,f"场景: UE{nUE}RB{nRB}, {test_num}次实验平均后结果","="*20)
    print("最终目标值：", np.mean(res))
    print("UE/RB配对数量:", np.mean(num_pair))
    print("="*20,"done:","="*20)
