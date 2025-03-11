这个实验结果阐述了一种bug情况,即:
当环境episode_length >= 所有可能的动作数量时,
mask会将全部动作都mask掉, 这样导致模型训练出现问题.
我们应该寻找一个合理的episode_length, 目前经验设计为 episode_length= 所有可能动作数量//4