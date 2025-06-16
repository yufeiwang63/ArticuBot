from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 
from einops.layers.torch import Rearrange
from termcolor import cprint
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy_3d.model.diffusion.conv1d_components import Conv1dBlock


logger = logging.getLogger(__name__)


class FilmConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class FilmConditionalResidualBlock1DSmall(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        return out
    
class FilmConditionalResidualBlockSmall(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FilmConditionalResidualBlock1D(in_channels,
                                             out_channels,
                                             cond_dim,
                                             kernel_size,
                                             n_groups),
        ])

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x, cond)
        return out


class FilmConditionalResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FilmConditionalResidualBlock1D(in_channels,
                                             out_channels * 4,
                                             cond_dim,
                                             kernel_size,
                                             n_groups),
            FilmConditionalResidualBlock1D(out_channels * 4,
                                                out_channels * 4,
                                                cond_dim,
                                                kernel_size,
                                                n_groups),
            FilmConditionalResidualBlock1D(out_channels * 4,
                                                out_channels,
                                                cond_dim,
                                                kernel_size,
                                                n_groups),
        ])

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x, cond)
        out = self.blocks[1](out, cond)
        out = self.blocks[2](out, cond)
        return out



class ConditionalTransformer(nn.Module):
    def __init__(self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        encoder_feature_dim=60,
        diffusion_attn_embed_dim=120,
        policy_type='high_level',
        **kwargs
    ):
        cprint("========= USING CONDITIONAL TRANSFORMER =========", "green")
        if local_cond_dim is not None or global_cond_dim is None:
            cprint("Only support global condition for ConditionalTransformer now", "red")
            cprint("Contact Optimus Prime to update Transformers to support local condition", "red")
            raise NotImplementedError
        
        # cprint("Only points action is supported for ConditionalTransformer now", "red")
        # assert input_dim == 12, "Only points action is supported for ConditionalTransformer now"
        self.policy_type = policy_type
        
        super().__init__()
        embedding_dim = diffusion_attn_embed_dim
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.relative_pe_emb = RotaryPositionEncoding3D(embedding_dim)

        self.observed_gripper_points_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        # self.observed_gripper_points_encoder = FilmConditionalResidualBlock1D(
        #     3, embedding_dim, global_cond_dim + embedding_dim
        # )

        if self.policy_type == 'high_level':
            self.sample_points_encoder = FilmConditionalResidualBlock(
                3, embedding_dim, global_cond_dim + embedding_dim
            )
        elif self.policy_type == 'low_level':
            self.sample_points_encoder = FilmConditionalResidualBlock(
                10, embedding_dim, global_cond_dim + embedding_dim
            )

        # self.scene_features_encoder = FilmConditionalResidualBlock1D(
        #     encoder_feature_dim, embedding_dim, global_cond_dim + embedding_dim
        # )

        self.obserseved_gripper_points_conditional_encoder = FilmConditionalResidualBlock(
            embedding_dim * 2, embedding_dim, global_cond_dim + embedding_dim
        )
        self.scene_conditional_encoder = FilmConditionalResidualBlock(
            embedding_dim + encoder_feature_dim, embedding_dim, global_cond_dim + embedding_dim
        )
        self.sample_points_conditional_encoder = FilmConditionalResidualBlock(
            embedding_dim * 2, embedding_dim, global_cond_dim + embedding_dim
        )

        self.cross_attn_obsered_gripper_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
        )

        self.cross_attn_scene_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
        )

        self.self_attn_conditional_layer = FilmConditionalResidualBlock(
            embedding_dim, embedding_dim, global_cond_dim + embedding_dim
        )

        self.self_attn_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
        )

        if self.policy_type == 'high_level':
            self.output_layer = FilmConditionalResidualBlock(
                embedding_dim, 3, global_cond_dim + embedding_dim
            )
        else:
            self.output_layer = FilmConditionalResidualBlock(
                embedding_dim, 10, global_cond_dim + embedding_dim
            )
            
            self.goal_gripper_points_encoder = nn.Sequential(nn.Linear(3, embedding_dim * 4),nn.ReLU(),nn.Linear(embedding_dim * 4, embedding_dim))
            
            self.goal_gripper_points_conditional_encoder = FilmConditionalResidualBlock(
                embedding_dim * 2, embedding_dim, global_cond_dim + embedding_dim
            )
            
            self.cross_attn_goal_layer = RelativeCrossAttentionModule(embedding_dim, 4, 3, hidden_dim=4*embedding_dim)



    
    def forward(self, 
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: Union[torch.Tensor],
                observed_gripper_points: Union[torch.Tensor],
                scene_points: Union[torch.Tensor],
                scene_features: Union[torch.Tensor],
                goal_gripper_points=None,
                num_gripper_points=4, 
                **kwargs):
        """
        Args:
            sample: (B, T, C) C = num_gripper_points * 3
            timestep: (B,)
            global_cond: (B, global_cond_dim)
            observed_gripper_points: (B, n_obs, num_gripper_points, 3)
            scene_points: (B, n_obs, N, 3)
            scene_features: (B, n_obs, N, encoder_feature_dim)
            goal_gripper_points: (B, n_obs, num_gripper_points, 3)
        """
        B, T, _ = sample.shape
        n_obs = observed_gripper_points.shape[1]
        n_scene_points = scene_points.shape[2]
        
        cur_gripper_position = observed_gripper_points[:, -1, -1, :] # the last point in the gripper pc is the eef position

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        timestep_emb = self.time_emb(timesteps) # B, embedding_dim
        global_feature = torch.cat([timestep_emb, global_cond], axis=-1) # B, embedding_dim + global_cond_dim

        # timestep embedding for observation steps
        observation_steps = torch.arange(n_obs, device=sample.device).float()
        obs_steps_embed = self.traj_time_emb(observation_steps) # n_obs, embedding_dim
        obs_steps_embed = obs_steps_embed[None].expand(B, -1, -1) # B, n_obs, embedding_dim

        observed_gripper_points_features = self.observed_gripper_points_encoder(observed_gripper_points) # B, n_obs, num_gripper_points, embedding_dim
        observed_gripper_points_features = torch.cat([observed_gripper_points_features, obs_steps_embed.unsqueeze(2).expand(-1, -1, num_gripper_points, -1)], dim=-1) # B, n_obs, num_gripper_points, embedding_dim * 2
        observed_gripper_points_features = einops.rearrange(
            observed_gripper_points_features, 'b n_obs N c -> b c (n_obs N)'
        )
        observed_gripper_points_features = self.obserseved_gripper_points_conditional_encoder(observed_gripper_points_features, global_feature) # B, embedding_dim, n_obs * num_gripper_points
        observed_gripper_points_features = einops.rearrange(
            observed_gripper_points_features, 'b c n_obs_x_N -> n_obs_x_N b c'
        ) # n_obs * num_gripper_points, B, embedding_dim

        observed_gripper_points = einops.rearrange(observed_gripper_points, 'b n_obs N c -> b (n_obs N) c') # B, n_obs * num_gripper_points, 3

        scene_features = torch.cat([scene_features, obs_steps_embed.unsqueeze(2).expand(-1, -1, n_scene_points, -1)], dim=-1) # B, n_obs, N, embedding_dim * 2
        scene_features = einops.rearrange(
            scene_features, 'b n_obs N c -> b c (n_obs N)'
        )
        scene_features = self.scene_conditional_encoder(scene_features, global_feature) # B, embedding_dim, n_obs * N
        scene_features = einops.rearrange(
            scene_features, 'b c n_obs_x_N -> n_obs_x_N b c'
        ) # n_obs * N, B, embedding_dim 
        scene_points = einops.rearrange(
            scene_points, 'b n_obs N c -> b (n_obs N) c'
        ) # B, n_obs * N, 3

        observed_gripper_points_pos_emb = self.relative_pe_emb(observed_gripper_points) # B, n_obs * num_gripper_points, embedding_dim, 2
        scene_pos_emb = self.relative_pe_emb(scene_points) # B, n_obs * N, embedding_dim, 2


        if self.policy_type == 'high_level':
            sample_points = sample.reshape(B, T, num_gripper_points, 3)
            sample_points_features = self.sample_points_encoder(sample_points.permute(0, 3, 1, 2).reshape(B, 3, T * num_gripper_points), global_feature) # B, embedding_dim, T * num_gripper_points
            sample_points_features = sample_points_features.permute(0, 2, 1).reshape(B, T, num_gripper_points, -1) # B, T, num_gripper_points, embedding_dim
            action_steps = torch.arange(T, device=sample.device).float()
            action_steps_embed = self.traj_time_emb(action_steps)
            sample_points_features = torch.cat([sample_points_features, action_steps_embed.unsqueeze(0).unsqueeze(2).repeat(B, 1, num_gripper_points, 1)], dim=-1) # B, T, num_gripper_points, embedding_dim * 2
            sample_points_features = einops.rearrange(
                sample_points_features, 'b T N c -> b c (T N)'
            )
            sample_points_features = self.sample_points_conditional_encoder(sample_points_features, global_feature) # (T N), B, embedding_dim
            sample_points_features = einops.rearrange(
                sample_points_features, 'b c T_x_N -> T_x_N b c'
            ) # T * N, B, embedding_dim
            sample_points_pos_emb = self.relative_pe_emb(sample_points.reshape(B, T * num_gripper_points, 3)) # B, T * num_gripper_points, embedding_dim, 2
            
            attn_output = self.cross_attn_obsered_gripper_layer(
                query=sample_points_features, value=observed_gripper_points_features,
                query_pos=sample_points_pos_emb, value_pos=observed_gripper_points_pos_emb
            )[-1] # (T N), B, embedding_dim
            #import pdb; pdb.set_trace()
            attn_output = self.cross_attn_scene_layer(
                query=attn_output, value=scene_features,
                query_pos=sample_points_pos_emb, value_pos=scene_pos_emb
            )[-1] # (T N), B, embedding_dim

            attn_output = einops.rearrange(
                attn_output, 'N B c -> B c N'
            )
            attn_output = self.self_attn_conditional_layer(attn_output, global_feature) # B, embedding_dim, T * num_gripper_points
            attn_output = einops.rearrange(
                attn_output, 'B c N -> N B c'
            )


            attn_output = self.self_attn_layer(
                query=attn_output, value=attn_output,
                query_pos=sample_points_pos_emb, value_pos=sample_points_pos_emb
            )[-1] # (T N), B, embedding_dim

            attn_output = einops.rearrange(
                attn_output, 'N B c -> B c N'
            )
            output = self.output_layer(attn_output, global_feature) # B, 3, T * num_gripper_points
            output = einops.rearrange(
                output, 'B c N -> B N c'
            ) # Output = 1, 16, 3
            #import pdb; pdb.set_trace()
            output = output.reshape(B, T, num_gripper_points, 3) 
            output = output.reshape(B, T, num_gripper_points * 3)
            
        elif self.policy_type == 'low_level':
            # sample: B, T, 10 (3d translation + 6d orientation + 1d finger movement)
            sample_points_features = self.sample_points_encoder(sample.permute(0, 2, 1), global_feature) # B, embedding_dim, T 
            sample_points_features = sample_points_features.permute(0, 2, 1) # B, T, embedding_dim
            action_steps = torch.arange(T, device=sample.device).float()
            action_steps_embed = self.traj_time_emb(action_steps)
            sample_points_features = torch.cat([sample_points_features, action_steps_embed.unsqueeze(0).repeat(B, 1, 1)], dim=-1) # B, T, embedding_dim * 2
            sample_points_features = einops.rearrange(sample_points_features, 'b T c -> b c T')
            sample_points_features = self.sample_points_conditional_encoder(sample_points_features, global_feature) # B, embedding_dim, T
            sample_points_features = einops.rearrange(sample_points_features, 'b c T -> T b c') # T, B, embedding_dim
            
            
            sample_delta_translation = sample[:, :, :3]
            after_delta_gripper_position = sample_delta_translation + cur_gripper_position.unsqueeze(1) # B, T, 3
            sample_points_pos_emb = self.relative_pe_emb(after_delta_gripper_position) # B, T, embedding_dim, 2
            
            ### cross attention between sample and observed gripper 
            attn_output = self.cross_attn_obsered_gripper_layer(query=sample_points_features, value=observed_gripper_points_features,query_pos=sample_points_pos_emb, value_pos=observed_gripper_points_pos_emb)[-1] # T, B, embedding_dim

            ### cross attention between sample and scene feature
            attn_output = self.cross_attn_scene_layer(query=attn_output, value=scene_features, query_pos=sample_points_pos_emb, value_pos=scene_pos_emb)[-1] # T, B, embedding_dim

            ### cross attention between sample and goal gripper 
            # preparing goal gripper features and position embeddings
            observation_steps = torch.arange(n_obs, device=sample.device).float()
            obs_steps_embed = self.traj_time_emb(observation_steps) # n_obs, embedding_dim
            obs_steps_embed = obs_steps_embed[None].expand(B, -1, -1) # B, n_obs, embedding_dim

            goal_gripper_points_features = self.goal_gripper_points_encoder(goal_gripper_points) # B, n_obs, num_gripper_points, embedding_dim
            goal_gripper_points_features = torch.cat([goal_gripper_points_features, obs_steps_embed.unsqueeze(2).expand(-1, -1, num_gripper_points, -1)], dim=-1) # B, n_obs, num_gripper_points, embedding_dim * 2
            goal_gripper_points_features = einops.rearrange(goal_gripper_points_features, 'b n_obs N c -> b c (n_obs N)')
            goal_gripper_points_features = self.goal_gripper_points_conditional_encoder(goal_gripper_points_features, global_feature) # B, embedding_dim, n_obs * num_gripper_points
            goal_gripper_points_features = einops.rearrange(goal_gripper_points_features, 'b c n_obs_x_N -> n_obs_x_N b c') # n_obs * num_gripper_points, B, embedding_dim

            goal_gripper_points = einops.rearrange(goal_gripper_points, 'b n_obs N c -> b (n_obs N) c') # B, n_obs * num_gripper_points, 3

            goal_gripper_points_pos_emb = self.relative_pe_emb(goal_gripper_points) # B, n_obs * num_gripper_points, embedding_dim, 2
            
            # cross attention
            attn_output = self.cross_attn_goal_layer(query=attn_output, value=goal_gripper_points_features, query_pos=sample_points_pos_emb, value_pos=goal_gripper_points_pos_emb)[-1]

            ### final self attention
            attn_output = einops.rearrange(attn_output, 'N B c -> B c N')
            attn_output = self.self_attn_conditional_layer(attn_output, global_feature) # B, embedding_dim, T
            attn_output = einops.rearrange(attn_output, 'B c N -> N B c')

            attn_output = self.self_attn_layer(query=attn_output, value=attn_output, query_pos=sample_points_pos_emb, value_pos=sample_points_pos_emb)[-1] # 10, B, embedding_dim

            attn_output = einops.rearrange(attn_output, 'N B c -> B c N')
            output = self.output_layer(attn_output, global_feature) # B, 10, T
            output = einops.rearrange(output, 'B c N -> B N c') 
            
        return output









        
        


        