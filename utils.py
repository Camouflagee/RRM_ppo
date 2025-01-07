import os, copy, pickle
from datetime import datetime
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
def save_model_env(log_dir: str, model_name: str, info: str, model_instance, env_instance,
                   reward=None):
    """
    Args:
        model_name: 模型的名字
        model_instance: 保存的模型实例
        env_instance: 保存的环境实例， 环境与模型需一一对应才可以eval
        :param log_dir:
        :param model_name:
        :param env_instance:
        :param model_instance:
        :param args: args of training the modules and initialization of the modules
    """
    # if start_time_str is None:
    #     now = datetime.datetime.now()
    #     time_str = now.strftime("date%Y%m%dtime%H%M%S")
    # else:
    #     time_str = start_time_str
    # file_t = "{}-{}.zip".format(method_name)
    # env_file_t = "{}-{}-env.zip".format(method_name, str_t)

    model_log_path = f'{log_dir}\\model_saves'
    env_path_with_name = f'{log_dir}\\model_saves\\env.zip'
    if not os.path.exists(model_log_path):
        os.makedirs(model_log_path)
    # os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    model_path_with_name = None
    if model_instance is not None:
        if reward is not None:
            model_path_with_name = model_log_path + f'\\{model_name}_NumSteps_{model_instance.num_timesteps}_reward{reward}.zip'
        else:
            model_path_with_name = model_log_path + f'\\{model_name}_NumSteps_{model_instance.num_timesteps}.zip'
        model_instance.save(path=model_path_with_name)
        print('modules was saved as {}'.format(model_path_with_name))
    if env_instance is not None:
        try:
            env_instance.configuration_visualization(log_dir + '\\model_saves\\env_plot.png')
        except Exception as e:
            print(e)
        with open(env_path_with_name, 'wb') as file:
            pickle.dump(env_instance, file, 0)
            print('env was saved in {}'.format(env_path_with_name))
    # only for stable-baseline3
    return model_path_with_name, env_path_with_name

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