import wandb
import numpy as np
import torch
import tqdm

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import os, json
from multiprocessing import Pool
from eval_robogen_parallel_new import run_eval

class ChainedDiffuserRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=1,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 num_worker=8,
                 **kwargs, 
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.save_video_dir = os.path.join(output_dir, 'videos')
        if not os.path.exists(self.save_video_dir):
            os.makedirs(self.save_video_dir, exist_ok=True)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.pool = Pool(processes=num_worker)
        self.num_worker = num_worker
        
       
    def run(self, cfg, policy: BasePolicy, epoch: int):
        num_worker = self.num_worker
        save_path = os.path.join(self.save_video_dir, f'epoch_{epoch}_trainset')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # exp_beg_idx = 0
        # exp_end_idx = cfg.task.dataset.num_load_episodes # cfg.task.dataset.train_ratio
        # [Chialiang]
        exp_beg_idx = int(cfg.task.dataset.num_load_episodes * 0.9)
        exp_end_idx = cfg.task.dataset.num_load_episodes # cfg.task.dataset.train_ratio
        # run_eval(cfg, policy, num_worker, save_path, pool=self.pool, horizon=self.max_steps, exp_beg_ratio=exp_beg_ratio, exp_end_ratio=exp_end_ratio)
        run_eval(cfg, policy, num_worker, save_path, pool=self.pool, horizon=self.max_steps, exp_beg_idx=exp_beg_idx, exp_end_idx=exp_end_idx)
        with open("{}/opened_joint_angles.json".format(save_path), "r") as f:
            result = json.load(f)
        train_all_success_rates = []
        for key in result:
            final_angle = result[key]["final_door_joint_angle"]
            intial_angle = result[key]["initial_joint_angle"]
            expert_angle = result[key]["expert_door_joint_angle"] if "46462" not in key else 0.27
            performance = (final_angle - intial_angle) / (expert_angle - intial_angle)
            performance = min(performance, 1)
            train_all_success_rates.append(performance)
            
        save_path = os.path.join(self.save_video_dir, f'epoch_{epoch}_valset')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        exp_beg_ratio = 1 - cfg.task.dataset.val_ratio
        exp_end_ratio = 1
        # run_eval(cfg, policy, num_worker, save_path, pool=self.pool, horizon=self.max_steps, exp_beg_ratio=exp_beg_ratio, exp_end_ratio=exp_end_ratio)
        # with open("{}/opened_joint_angles.json".format(save_path), "r") as f:
        #     result = json.load(f)
        # val_all_success_rates = []
        # for key in result:
        #     final_angle = result[key][0]
        #     intial_angle = result[key][2]
        #     val_all_success_rates.append(final_angle - intial_angle)

        # log
        log_data = dict()
        

        # log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(train_all_success_rates)
        log_data['trainset_mean_success_rates'] = np.mean(train_all_success_rates)
        # log_data['valset_mean_success_rates'] = np.mean(val_all_success_rates)

        log_data['test_mean_score'] = np.mean(train_all_success_rates)
        log_data['trainset_test_mean_score'] = np.mean(train_all_success_rates)
        # log_data['valset_test_mean_score'] = np.mean(val_all_success_rates)

        cprint(f"trainset test_mean_score: {np.mean(train_all_success_rates)}", 'green')
        # cprint(f"valset test_mean_score: {np.mean(val_all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(train_all_success_rates))
        self.logger_util_test10.record(np.mean(train_all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb
        
        with open(os.path.join(self.save_video_dir, f'eval_trainset_results.txt'), 'a') as f:
            f.write(str(np.mean(train_all_success_rates)) + '\n')

        return log_data
