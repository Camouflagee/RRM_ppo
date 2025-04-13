import numpy as np
from gymnasium.wrappers import TimeLimit
from policy.sequence_ppo import SequencePPO
from utils import load_env
import os
import re
import warnings
from utils import Logger

warnings.filterwarnings("ignore")


def get_model_paths(root_dir):
    """获取所有训练好的模型路径"""
    model_paths = []
    error_rates = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.zip') and 'model_saves' in root:
                # if file.endswith('.zip') and 'model_saves' in root and 'best' in file:
                model_paths.append(os.path.join(root, file))
                # 从路径中提取error rate
                error_match = re.search(r'error_([0-9.]+)', root)
                error_rate = float(error_match.group(1)) if error_match else 0.0
                error_rates.append(error_rate)
    return model_paths, error_rates


def eval_model(model_path, error_rate, use_sideinfo, given_obs=None):
    """评估单个模型的性能"""
    nUE = 12
    nRB = 30
    Nrb = 15
    episode_length = nUE * Nrb
    res = []
    num_pair = []
    test_num = 80

    # 加载环境和模型
    unwrapped_env = load_env(f'Experiment_result/seqPPOcons_R2A3_sideinfo/UE{nUE}RB{nRB}/ENV/env.zip')
    model = SequencePPO.load(model_path)

    # 设置环境参数
    unwrapped_env.error_percent = error_rate
    unwrapped_env.use_sideinfo = use_sideinfo
    unwrapped_env.eval_mode = True
    test_env = TimeLimit(unwrapped_env, max_episode_steps=episode_length)

    # 测试循环
    for _ in range(test_num):
        obs, _ = test_env.reset_onlyforbaseline(given_obs)
        truncated = False
        while not truncated:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if truncated:
                res.append(reward)
                num_pair.append(sum(obs[nUE * nRB:]))

    return np.mean(res), np.mean(num_pair)

    # def main():
    # 测试使用side info的模型
    # print("\n====== 测试使用side info的模型 ======")
    # model_paths, error_rates = get_model_paths("Experiment_result/seqPPOcons_R2A3_sideinfo")
    # res_si=[[],[]]
    # for model_path, error_rate in zip(model_paths, error_rates):
    #     reward, pairs = eval_model(model_path, error_rate, True)
    #     print(f"\n错误率: {error_rate:.2f}")
    #     # print(f"模型路径: {model_path}")
    #     print(f"平均奖励: {reward:.3f}")
    #     print(f"平均配对数: {pairs:.3f}")
    #     res_si[0].append(reward)
    #     res_si[1].append(pairs)
    # # 测试不使用side info的模型
    # print("\n====== 测试不使用side info的模型 ======")
    # model_paths, error_rates = get_model_paths("Experiment_result/seqPPOcons_R2A3_nosideinfo")
    # res_nosi=[[],[]]
    # for model_path, error_rate in zip(model_paths, error_rates):
    #     reward, pairs = eval_model(model_path, error_rate, False)
    #     print(f"\n错误率: {error_rate:.2f}")
    #     # print(f"模型路径: {model_path}")
    #     print(f"平均奖励: {reward:.3f}")
    #     print(f"平均配对数: {pairs:.3f}")
    #     res_nosi[0].append(reward)
    #     res_nosi[1].append(pairs)
    # # 输出结果
    # print("\n====== 测试结果 ======")
    # print("使用side info的模型:")
    # print("平均奖励: ",res_si[0])
    # print("平均配对数: ",res_si[1])
    # print("不使用side info的模型:")
    # print("平均奖励: ",res_nosi[0])
    # print("平均配对数: ", res_nosi[1])

    # def main():
    # 创建logger实例
    # with Logger("Experiment_result/baseline/eval_results.txt") as logger:
    #     # 测试使用side info的模型
    #     logger.log("\n====== 测试使用side info的模型 ======")
    #     model_paths, error_rates = get_model_paths("Experiment_result/seqPPOcons_R2A3_sideinfo")
    #     res_si = [[], []]
    #     best_res_si = [[], []]
    #     for idx, (model_path, error_rate) in enumerate(zip(model_paths, error_rates)):
    #         reward, pairs = eval_model(model_path, error_rate, True)
    #         logger.log(f"\n错误率: {error_rate:.2f}")
    #         logger.log(f"平均奖励: {reward:.3f}")
    #         logger.log(f"平均配对数: {pairs:.3f}")
    #         res_si[0].append(reward)
    #         res_si[1].append(pairs)
    #         if idx % 2 == 0:
    #             best_res_si[0].append(max(res_si[0][-2:]))
    #             best_res_si[1].append(max(res_si[1][-2:]))
    #
    #     # 测试不使用side info的模型
    #     logger.log("\n====== 测试不使用side info的模型 ======")
    #     model_paths, error_rates = get_model_paths("Experiment_result/seqPPOcons_R2A3_nosideinfo")
    #     res_nosi = [[], []]
    #     best_res_nosi = [[], []]
    #     for idx, (model_path, error_rate) in enumerate(zip(model_paths, error_rates)):
    #         reward, pairs = eval_model(model_path, error_rate, False)
    #         logger.log(f"\n错误率: {error_rate:.2f}")
    #         logger.log(f"平均奖励: {reward:.3f}")
    #         logger.log(f"平均配对数: {pairs:.3f}")
    #         best_res_nosi[0].append(reward)
    #         best_res_nosi[1].append(pairs)
    #         if idx % 2 == 0:
    #             best_res_nosi[0].append(max(res_nosi[0][-2:]))
    #             best_res_nosi[1].append(max(res_nosi[1][-2:]))
    #     # 输出结果
    #     logger.log("\n====== 测试结果 ======")
    #     logger.log("使用side info的模型:")
    #     logger.log(f"平均奖励: {best_res_si[0]}")
    #     logger.log(f"平均配对数: {best_res_si[1]}")
    #     logger.log("不使用side info的模型:")
    #     logger.log(f"平均奖励: {best_res_nosi[0]}")
    #     logger.log(f"平均配对数: {best_res_nosi[1]}")


def evaluate_models(logger, model_dir, use_sideinfo):
    """评估指定目录下的所有模型"""
    model_paths, error_rates = get_model_paths(model_dir)
    results = [[], []]
    best_results = [[], []]

    for idx, (model_path, error_rate) in enumerate(zip(model_paths, error_rates)):
        reward, pairs = eval_model(model_path, error_rate, use_sideinfo)
        logger.log(f"\n错误率: {error_rate:.2f}")
        logger.log(f"平均奖励: {reward:.3f}")
        logger.log(f"平均配对数: {pairs:.3f}")
        results[0].append(reward)
        results[1].append(pairs)
        if idx % 2 == 0:
            best_results[0].append(max(results[0][-2:]))
            best_results[1].append(max(results[1][-2:]))

    return best_results, results


def main():
    with Logger("Experiment_result/baseline/eval_res.txt") as logger:
        # 测试使用side info的模型
        logger.log("\n====== 测试使用side info的模型 ======")
        best_res_si, res_si = evaluate_models(logger, "Experiment_result/seqPPOcons_R2A3_sideinfo", True)

        # 测试不使用side info的模型
        logger.log("\n====== 测试不使用side info的模型 ======")
        best_res_nosi, res_nosi = evaluate_models(logger, "Experiment_result/seqPPOcons_R2A3_nosideinfo", False)

        # 输出结果
        logger.log("\n====== 测试结果 ======")
        logger.log("使用side info的模型:")
        logger.log(f"平均奖励: {best_res_si[0]}")
        logger.log(f"平均配对数: {best_res_si[1]}")
        logger.log("不使用side info的模型:")
        logger.log(f"平均奖励: {best_res_nosi[0]}")
        logger.log(f"平均配对数: {best_res_nosi[1]}")


if __name__ == "__main__":
    main()
