from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 
from einops.layers.torch import Rearrange
from termcolor import cprint
# from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule, FFWRelativeSelfCrossAttentionModule, FFWRelativeSelfAttentionModule
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



class ConditionalTransformer_Self_Attention(nn.Module):
    def __init__(self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        encoder_feature_dim=60,
        diffusion_attn_embed_dim=120,
        use_cross_self_attention=False,
        use_self_attention=False,
        **kwargs
    ):
        cprint("========= USING CONDITIONAL TRANSFORMER =========", "green")
        if local_cond_dim is not None or global_cond_dim is None:
            cprint("Only support global condition for ConditionalTransformer now", "red")
            cprint("Contact Optimus Prime to update Transformers to support local condition", "red")
            raise NotImplementedError
        
        cprint("Only points action is supported for ConditionalTransformer now", "red")
        assert input_dim == 12, "Only points action is supported for ConditionalTransformer now"
        
        super().__init__()
        self.use_cross_self_attention = False
        self.use_self_attention = True
        embedding_dim = diffusion_attn_embed_dim
        self.embedding_dim = diffusion_attn_embed_dim
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

        self.sample_points_encoder = FilmConditionalResidualBlock(
            3, embedding_dim, global_cond_dim + embedding_dim
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
        if self.use_cross_self_attention:
            self.cross_self_attn_obsered_gripper_layer = FFWRelativeSelfCrossAttentionModule(
            embedding_dim=embedding_dim, num_attn_heads=4, num_self_attn_layers=2, num_cross_attn_layers=3
            )
        elif self.use_self_attention:
            self.self_attn_obsered_gripper_layer = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads=4, num_layers=2
            )
        else:
            self.cross_attn_obsered_gripper_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
            )
        if self.use_cross_self_attention:
            self.cross_self_attn_scene_layer = FFWRelativeSelfCrossAttentionModule(
            embedding_dim=embedding_dim, num_attn_heads=4, num_self_attn_layers=2, num_cross_attn_layers=3
            )
        elif self.use_self_attention:
            self.self_attn_scene_layer = FFWRelativeSelfAttentionModule(
            embedding_dim=embedding_dim, num_attn_heads=4, num_layers=2
            )
        else:
            self.cross_attn_scene_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
            )

        self.self_attn_conditional_layer = FilmConditionalResidualBlock(
            embedding_dim, embedding_dim, global_cond_dim + embedding_dim
        )
        
        self.self_attn_layer = RelativeCrossAttentionModule(
            embedding_dim, 4, 3, hidden_dim=4*embedding_dim
        )

        self.output_layer = FilmConditionalResidualBlock(
            embedding_dim, 3, global_cond_dim + embedding_dim
        )


    
    def forward(self, 
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: Union[torch.Tensor],
                observed_gripper_points: Union[torch.Tensor],
                scene_points: Union[torch.Tensor],
                scene_features: Union[torch.Tensor],
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
        """
        #import pdb; pdb.set_trace()
        B, T, _ = sample.shape
        n_obs = observed_gripper_points.shape[1]
        n_scene_points = scene_points.shape[2]

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

        observed_gripper_points = einops.rearrange(
            observed_gripper_points, 'b n_obs N c -> b (n_obs N) c'
        ) # B, n_obs * num_gripper_points, 3

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
        if self.use_cross_self_attention:
            attn_output = self.cross_self_attn_obsered_gripper_layer(
                query=sample_points_features, context=observed_gripper_points_features,
                query_pos=sample_points_pos_emb, context_pos=observed_gripper_points_pos_emb
            )[-1] # (T N), B, embedding_dim
        elif self.use_self_attention:
            #import pdb; pdb.set_trace()
            gripper_shape = sample_points_pos_emb.shape[1]
            features = torch.cat([sample_points_features, observed_gripper_points_features], 0)
            rel_pos = torch.cat([sample_points_pos_emb, observed_gripper_points_pos_emb], 1)
            #print("USE SELF ATTN OBSERVED GRIPPER LAYER")
            #print("HEREEEEEEEEEEEEEEEEEEEEEE self attn observed gripper layer ATTN OUTPUT SHAPE", features.shape)
            attn_output = self.self_attn_obsered_gripper_layer(
            query=features,
            query_pos=rel_pos,
            diff_ts=None,
            context=None,
            context_pos=None
            )[-1]
            attn_output = attn_output[:gripper_shape, :, :]
        else:
            #print("HEREEEEEEEEEEEEEEEEEEEEEE cross attn observed gripper layer ATTN OUTPUT SHAPE", sample_points_features.shape)
            attn_output = self.cross_attn_obsered_gripper_layer(
                query=sample_points_features, value=observed_gripper_points_features,
                query_pos=sample_points_pos_emb, value_pos=observed_gripper_points_pos_emb
            )[-1] # (T N), B, embedding_dim
        if self.use_cross_self_attention:
            attn_output = self.cross_self_attn_scene_layer(
            query=attn_output, context=scene_features,
            query_pos=sample_points_pos_emb, context_pos=scene_pos_emb
            )[-1] # (T N), B, embedding_dim
        elif self.use_self_attention:
            gripper_shape = sample_points_pos_emb.shape[1]
            features = torch.cat([attn_output, scene_features], 0)
            rel_pos = torch.cat([sample_points_pos_emb, scene_pos_emb], 1)
            #print("USE SELF ATTN SCENE LAYER")
            attn_output = self.self_attn_scene_layer(
            query=features,
            query_pos=rel_pos,
            diff_ts=None,
            context=None,
            context_pos=None
            )[-1]
            attn_output = attn_output[:gripper_shape, :, :]
        else:
            attn_output = self.cross_attn_scene_layer(
            query=attn_output, value=scene_features,
            query_pos=sample_points_pos_emb, value_pos=scene_pos_emb
        )[-1] # (T N), B, embedding_dim

        attn_output = einops.rearrange(
            attn_output, 'N B c -> B c N'
        )
        #print("HEREEEEEEEEEEEEEEEEEEEEEE CONDITIONAL ATTN OUTPUT SHAPE", attn_output.shape)
        attn_output = self.self_attn_conditional_layer(attn_output, global_feature) # B, embedding_dim, T * num_gripper_points
        attn_output = einops.rearrange(
            attn_output, 'B c N -> N B c'
        )

        #print("USE SELF ATTN LAYER")
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
        )
        import pdb; pdb.set_trace()
        output = output.reshape(B, T, num_gripper_points, 3)
        output = output.reshape(B, T, num_gripper_points * 3)
        return output









        
        


        