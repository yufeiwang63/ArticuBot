from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.transformers.create_transformer import create_conditional_transformer
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.vision.act3d_encoder import Act3dEncoder
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn

class DP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True, # true
            use_mid_condition=True, # true
            use_up_condition=True, # true
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            use_state=True,
            encoder_type='pointnet',
            act3d_encoder_cfg=None,
            prediction_target='action',
            noise_model_type='unet',
            diffusion_attn_embed_dim=120,
            transformer_type = "default",
            normalize_action=True, 
            scale_scene_by_pcd=False, 
            policy_type='high_level',
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.prediction_target = prediction_target
        self.normalize_action = normalize_action
        self.scale_scene_by_pcd = scale_scene_by_pcd
        self.act3d_encoder_cfg = act3d_encoder_cfg

        # parse shape_meta
        action_shape = shape_meta[self.prediction_target]['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        self.encoder_type = encoder_type
        if self.encoder_type=="dp3":
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                use_state=use_state
                                                )
            # create diffusion model
            obs_feature_dim = obs_encoder.output_shape()
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
            if obs_as_global_cond:
                input_dim = action_dim
                if "cross_attention" in self.condition_type:
                    global_cond_dim = obs_feature_dim
                else:
                    global_cond_dim = obs_feature_dim * n_obs_steps
                    
            self.use_pc_color = use_pc_color
            self.pointnet_type = pointnet_type
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
                    
        elif self.encoder_type == 'act3d':
            obs_encoder = Act3dEncoder(**act3d_encoder_cfg, encoder_output_dim=encoder_output_dim, 
                                       observation_space=obs_dict)

            obs_feature_dim = obs_encoder.output_shape()
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
            if obs_as_global_cond:
                input_dim = action_dim
                if "cross_attention" in self.condition_type:
                    global_cond_dim = obs_feature_dim
                else:
                    global_cond_dim = obs_feature_dim * n_obs_steps

        self.encoder_output_dim = encoder_output_dim
        self.noise_model_type = noise_model_type
        if self.noise_model_type == 'unet':
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )
        elif self.noise_model_type == 'transformer':
            model = create_conditional_transformer(transformer_type = transformer_type,
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                encoder_feature_dim=encoder_output_dim,
                diffusion_attn_embed_dim=diffusion_attn_embed_dim,
                policy_type=policy_type,
            )

        if self.encoder_type == "act3d":
            model = replace_bn_with_gn(model)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        cprint(f'using {self.noise_scheduler.config.prediction_type}', 'green')

        print_params(self)
        cprint('model has been loaded', 'green')
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond, **kwargs)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        if "act3d" not in self.encoder_type:
            nobs = {
                "point_cloud": obs_dict['point_cloud'],
                "gripper_pcd": obs_dict['gripper_pcd'],
                "agent_pos": obs_dict['agent_pos'],
            }
            nobs = self.normalizer.normalize(nobs)
        else:
            nobs = obs_dict

        if self.scale_scene_by_pcd:
            max_scale = torch.max(torch.norm(nobs['point_cloud'][...,:3], dim=-1))

            nobs['point_cloud'][...,:3] /= max_scale
            nobs['agent_pos'][...,:3] /= max_scale

            if "act3d" in self.encoder_type:
                nobs['gripper_pcd'][...,:3] /= max_scale
                if 'goal' in self.act3d_encoder_cfg:
                    nobs['goal_gripper_pcd'][...,:3] /= max_scale
                if 'displacement_gripper_to_object' in self.act3d_encoder_cfg:
                    nobs['displacement_gripper_to_object'][...,:3] /= max_scale
            
            elif 'act3d_pointnet' == self.act3d_encoder_cfg:
                nobs['gripper_pcd'][...,:3]  /= max_scale

        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        if self.obs_as_global_cond:
            # condition through global feature
            # reshape from Batch_size, horizon, ... to Batch_size*horizon, ...
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        if self.noise_model_type == 'unet':
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
        elif self.noise_model_type == 'transformer':
            observed_gripper_points = this_nobs['gripper_pcd'].reshape(B, self.n_obs_steps, -1, 3)
            scene_points, scene_features = self.obs_encoder.get_pcd_features()
            scene_points = scene_points.reshape(B, self.n_obs_steps, -1, 3)
            scene_features = scene_features.reshape(scene_points.shape[2], B, self.n_obs_steps, self.encoder_output_dim)
            scene_features = scene_features.permute(1, 2, 0, 3)
            nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond, observed_gripper_points=observed_gripper_points, scene_points=scene_points, scene_features=scene_features, goal_gripper_points=this_nobs['goal_gripper_pcd'].reshape(B, self.n_obs_steps, -1, 3), **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        if self.prediction_target == 'action' or self.prediction_target == 'delta_to_goal_gripper':
            action_pred = naction_pred
            if self.normalize_action:
                action_pred_backup = copy.deepcopy(action_pred.detach())
                action_pred = self.normalizer[self.prediction_target].unnormalize(action_pred)
                
                # for rotation augmentation only
                if 'additional_params' in self.normalizer.params_dict.keys():
                    max_norm_3d = self.normalizer.params_dict['additional_params']['max_norm_3d'][0]
                    # for unnormalizing delta position
                    action_pred[...,:3] = action_pred_backup[...,:3] * max_norm_3d
                    # for delta rotation, actually no need to normalize because the original 6D representation ensure they will be in [-1, 1]
                    action_pred[...,3:9] = action_pred_backup[...,3:9]
                    # for delta gripper pose (unchanged, so no operation)

        else:
            action_pred = naction_pred

        if self.scale_scene_by_pcd:
            action_pred[...,:3] *= max_scale

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        # normalize input

        if 'act3d' not in self.encoder_type:
            nobs = {
                "point_cloud": batch['obs']['point_cloud'],
                "gripper_pcd": batch['obs']['gripper_pcd'],
                "agent_pos": batch['obs']['agent_pos'],
            }
            nobs = self.normalizer.normalize(nobs)

        else:
            nobs = batch['obs']
        
        if  self.prediction_target == 'action' or self.prediction_target == 'delta_to_goal_gripper':
            if self.prediction_target == 'action':
                nactions = batch[self.prediction_target]
            elif self.prediction_target == 'delta_to_goal_gripper':
                nactions = batch['obs'][self.prediction_target].flatten(start_dim=2)
            
            if self.normalize_action:
                nactions_backup = copy.deepcopy(nactions)
                nactions = self.normalizer[self.prediction_target].normalize(nactions)

                # for rotation augmentation only
                if 'additional_params' in self.normalizer.params_dict.keys():
                    max_norm_3d = self.normalizer.params_dict['additional_params']['max_norm_3d'][0]
                    # for unnormalizing delta position
                    nactions[...,:3] = nactions_backup[...,:3] / max_norm_3d
                    # for delta rotation, actually no need to normalize because the original 6D representation ensure they will be in [-1, 1]
                    nactions[...,3:9] = nactions_backup[...,3:9]
                    # for delta gripper pose (unchanged, so no operation)

        else:
            nactions = batch['obs'][self.prediction_target].flatten(start_dim=2)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)


            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        if self.noise_model_type == 'unet':
            pred = self.model(sample=noisy_trajectory, 
                            timestep=timesteps, 
                                local_cond=local_cond, 
                                global_cond=global_cond)
        elif self.noise_model_type == 'transformer':
            observed_gripper_points = this_nobs['gripper_pcd'].reshape(batch_size, self.n_obs_steps, -1, 3)
            scene_points, scene_features = self.obs_encoder.get_pcd_features()
            scene_points = scene_points.reshape(batch_size, self.n_obs_steps, -1, 3)
            scene_features = scene_features.reshape(scene_points.shape[2], batch_size, self.n_obs_steps, self.encoder_output_dim)
            scene_features = scene_features.permute(1, 2, 0, 3)
            pred = self.model(sample=noisy_trajectory,
                                timestep=timesteps,
                                global_cond=global_cond,
                                observed_gripper_points=observed_gripper_points,
                                scene_points=scene_points,
                                scene_features=scene_features,
                                goal_gripper_points=this_nobs['goal_gripper_pcd'].reshape(batch_size, self.n_obs_steps, -1, 3))


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_ddim.py
            v_t = self.noise_scheduler.get_velocity(sample=trajectory, noise=noise, timesteps=timesteps)
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }

        return loss, loss_dict

