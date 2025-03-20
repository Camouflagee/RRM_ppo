import os, copy, pickle
from datetime import datetime

from environment import Environment


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


def save_model_env(log_dir: str, model_name: str, info: str, model_instance, env_instance: Environment):
    """
    Save model and env # only for stable-baseline3
    :param log_dir: time level log_dir e.g. 'Experiment_result\\seqPPOcons\\UE5RB10\\E1\\date20250311time121613',
    :param model_name: seqPPOcons
    :param info: some info added into the tail of file_path
    :param model_instance: to be saved
    :param env_instance: to be saved
    :return:
    """
    model_path_with_name, env_path_with_name = None, None
    if model_instance is not None:
        model_log_path = os.path.join(log_dir, 'model_saves')
        os.makedirs(model_log_path, exist_ok=True)
        if info is not None:
            model_path_with_name = os.path.join(model_log_path,
                                                f'{model_name}_NumSteps_{model_instance.num_timesteps}_{info}.zip')
        else:
            model_path_with_name = os.path.join(model_log_path,
                                                f'{model_name}_NumSteps_{model_instance.num_timesteps}.zip')
        model_instance.save(path=model_path_with_name)
        print('modules was saved as {}'.format(model_path_with_name))
    if env_instance is not None:
        time_str=os.path.split(log_dir)[-1]
        for _ in range(2):  # get the upper dir of upper dir
            env_dir = os.path.dirname(log_dir)
            log_dir = env_dir
        env_dir = os.path.join(env_dir, 'ENV')
        env_path_with_name = os.path.join(env_dir, f'env_{time_str}.zip')
        os.makedirs(env_dir, exist_ok=True)
        env_instance.showmap(os.path.join(env_dir, f'env_plot_{time_str}.png'))
        with open(env_path_with_name, 'wb') as file:
            pickle.dump(env_instance, file, 0)
            print('Env was saved in {}'.format(env_path_with_name))
    return model_path_with_name, env_path_with_name
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # 保存原始的标准输出流
        self.filename = filename
        self.prepare_log_file()  # 检查并准备日志文件
        self.log = open(filename, "a", encoding="utf-8")  # 以追加模式打开文件

    def prepare_log_file(self):
        """检查日志文件是否存在，存在则添加分割线"""
        if os.path.exists(self.filename):
            with open(self.filename, "a", encoding="utf-8") as file:
                file.write("\n======== 新日志分割线 ========\n")  # 添加分割线

    def write(self, message):
        self.terminal.write(message)  # 控制台输出
        self.log.write(message)  # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        """关闭日志文件"""
        self.log.close()
# # 设置日志文件名
# log_filename = "output_log.txt"
#
# # 将标准输出重定向到 Logger
# sys.stdout = Logger(log_filename)
#
# # 示例打印
# print("这是一个测试信息！")
# print("所有的print输出都会被记录到output_log.txt中。")

def get_TimeLogEvalDir(log_root: str = 'Experiment_result', model_name: str = 'FACMAC', args='args1',
                       time_str: str = None, info_str: str = ''):
    """
    return time level log_dir e.g. 'Experiment_result\\seqPPOcons\\UE5RB10\\E1\\date20250311time121613_{info_str}',
    :param model_name: seqPPOcons
    :param log_root: Experiment_result
    :param model_name: seqPPOcons
    :param args: UE5RB10\\E1
    :param time_str: date20250311time121613
    :param info_str: added in to the tail of time_dir
    """

    if time_str is None:
        now = datetime.now()
        time_str = now.strftime("date%Y%m%dtime%H%M%S")
    # f"./{log_root}/{model_name}/{args}/{time_str}" + info_str
    log_folder = os.path.join(log_root, model_name, args, time_str + info_str)
    eval_log_dir = os.path.join(log_root, model_name, args, time_str + info_str, 'model_saves', 'eval_best_model')
    # f'./{log_root}/{model_name}/{args}/{time_str}' + info_str + '/model_saves/eval_best_model/'

    return log_folder, eval_log_dir


def load_env(env_path):
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
    return env


def system_shutdown(time_to_be_shutdown: int = 240):
    import tkinter as tk
    import os
    def shutdown():
        os.system(f"shutdown /s /t {time_to_be_shutdown}")
        print('shutdown!')

    def cancel_shutdown():
        os.system("shutdown -a")
        print('cancel_shutdown!')
        window.destroy()

    # 创建窗口
    window = tk.Tk()
    window.geometry("300x100")
    window.title("Cancel shutdown countdown?")
    shutdown()
    # 在窗口显示时开始倒计时
    button_cancel = tk.Button(window, text="Yes :D", command=cancel_shutdown, width=10, height=2)
    button_cancel.pack()
    button_cancel.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    # 运行窗口的主循环
    window.mainloop()
