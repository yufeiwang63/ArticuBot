import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import einops
import copy
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D 
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp

class Act3dEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 encoder_output_dim=256, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 goal_mode=None,
                 mode=None,
                 use_mlp=True,
                 self_attention=False,
                 pointcloud_backbone='mlp',
                 final_attention=False,
                 attention_num_heads=3,
                 attention_num_layers=2,
                 **kwargs
                 ):
        super(Act3dEncoder, self).__init__()
        
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.gripper_pcd_key = 'gripper_pcd'
        self.num_gripper_points = num_gripper_points
        self.encoder_output_dim = encoder_output_dim
        self.state_shape = observation_space[self.state_key]
        self.goal_mode = goal_mode
        self.use_mlp = use_mlp
        
        self.self_attention = self_attention
        self.final_attention = final_attention
        self.mode = mode
        if self.mode in ['keep_position_feature_in_attention_feature']:
            vision_output_dim = encoder_output_dim // 3 * 2
        else:
            vision_output_dim = encoder_output_dim
        
        vision_encoder = None

        self.pointcloud_backbone = pointcloud_backbone
        if self.use_mlp:
            self.pointcloud_backbone = 'mlp'
        cprint("Using pointcloud backbone: " + self.pointcloud_backbone, 'green')

        if self.pointcloud_backbone == 'mlp':
            hidden_layer_dim = encoder_output_dim
            vision_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, encoder_output_dim)
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            cprint(f"Unknown pointcloud backbone {self.pointcloud_backbone}", 'red')
            
        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        attn_layers = replace_bn_with_gn(attn_layers)
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if self.self_attention:
            self.nets['self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['self_attn_layers'] = replace_bn_with_gn(self.nets['self_attn_layers'])
        
        input_dim = 3
        if self.goal_mode is not None:
            input_dim += 3
        if self.mode == "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object":
            input_dim += 3
        position_embedding_mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        object_pcd_position_embedding_mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, encoder_output_dim // 3),
        )
        self.nets['object_pcd_position_embedding_mlp'] = object_pcd_position_embedding_mlp
        self.nets['gripper_pcd_position_embedding_mlp'] = position_embedding_mlp
        self.nets['embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        # NOTE: 
        # here is how the low-level policy works:
        # cross attention between current gripper and object -> vec1
        # cross attention between current gripper and target gripper -> vec2
        # concatenate vectors -> diffusion -> output
        
        assert self.goal_mode == 'cross_attention_to_goal', "goal_mode must be 'cross_attention_to_goal' for Act3dEncoder, got {}".format(self.goal_mode)
        goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
        goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
        self.nets['goal_attn_layers'] = goal_attn_layers
        if self.mode in ['keep_position_feature_in_attention_feature', "keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object"]:
            self.nets['goal_pcd_position_embedding_mlp'] = copy.deepcopy(position_embedding_mlp)
            self.nets['goal_embed'] = nn.Embedding(1, encoder_output_dim // 3 * 2)

        if self.self_attention: ### add more self attention layers
            self.nets['goal_self_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)   # [Debug] make it deeper
            self.nets['goal_self_attn_layers'] = replace_bn_with_gn(self.nets['goal_self_attn_layers'])

        if self.final_attention: ### add more self attention layers
            self.nets['final_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['final_attn_layers'] = replace_bn_with_gn(self.nets['final_attn_layers'])
            self.nets['final_slef_attn_layers'] = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self.nets['final_slef_attn_layers'] = replace_bn_with_gn(self.nets['final_slef_attn_layers'])
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        if self.goal_mode == 'cross_attention_to_goal' and not self.final_attention: 
            self.n_output_channels += encoder_output_dim * self.num_gripper_points
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict, return_full=False) -> torch.Tensor:
        # NOTE: the things passed in is already flattend from B, T, ... -> B*T, ...
        nets = self.nets
        
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        pcd = observation[self.point_cloud_key]
        B, N, C = pcd.shape
        pcd_obs_flatten = pcd.reshape(-1, C)
        pcd_features_flatten = nets['vision_encoder'](pcd_obs_flatten)
        pcd_features = pcd_features_flatten.reshape(B, N, -1) # shape B N encoder_output_dim
        pcd_features = einops.rearrange(pcd_features, "B N encoder_output_dim -> N B encoder_output_dim") # shape N B encoder_output_dim
        
        point_cloud = observation[self.point_cloud_key]
        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        
        displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
        input_to_position_embedding = torch.cat([gripper_pcd, displacement_to_goal], dim=-1)
        if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
            displacement_to_closest_object = observation['displacement_gripper_to_object']
            input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
        input_to_position_embedding = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
        gripper_pcd_position_embedding = nets['gripper_pcd_position_embedding_mlp'](input_to_position_embedding)
        gripper_pcd_position_embedding = einops.rearrange(gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=num_gripper_points)
        gripper_pcd_features = torch.cat([gripper_pcd_features, gripper_pcd_position_embedding], dim=-1)

        self._pcd_features = pcd_features
        self._point_cloud = point_cloud
        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=pcd_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]
        
        if not self.self_attention:
            pcd_features = einops.rearrange(
                attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)
        else:
            self_attn_output = nets['self_attn_layers'](
                query=attn_output, value=attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]
            pcd_features = einops.rearrange(
                self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
            
        state_feat = self.state_mlp(agent_pos)  # B * 64
        obs_features = torch.cat([pcd_features, state_feat], dim=-1)

        goal_gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](observation['goal_gripper_pcd']) # shape B num_gripper_points encoder_output_dim
        goal_gripper_pcd_features = nets['goal_embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)
        displacement_to_goal = observation['goal_gripper_pcd'] - observation['gripper_pcd']
        input_to_position_embedding = torch.cat([observation['goal_gripper_pcd'], displacement_to_goal], dim=-1)
        if self.mode == 'keep_position_feature_in_attention_feature_with_gripper_displacement_to_closest_object':
            displacement_to_closest_object = observation['displacement_gripper_to_object']
            input_to_position_embedding = torch.cat([input_to_position_embedding, displacement_to_closest_object], dim=-1)
        goal_gripper_pcd_position = einops.rearrange(input_to_position_embedding, "B num_gripper_points c -> (B num_gripper_points) c", B=B, num_gripper_points=self.num_gripper_points)
        goal_gripper_pcd_position_embedding = nets['goal_pcd_position_embedding_mlp'](goal_gripper_pcd_position)
        goal_gripper_pcd_position_embedding = einops.rearrange(goal_gripper_pcd_position_embedding, "(B num_gripper_points) encoder_output_dim -> num_gripper_points B encoder_output_dim", B=B, num_gripper_points=self.num_gripper_points)
        goal_gripper_pcd_features = torch.cat([goal_gripper_pcd_features, goal_gripper_pcd_position_embedding], dim=-1)
                
        goal_attn_output = nets['goal_attn_layers'](query=gripper_pcd_features, value=goal_gripper_pcd_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
        )[-1]
        
        if self.self_attention:
            goal_attn_output = nets['goal_self_attn_layers'](query=goal_attn_output, value=goal_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]

        
        if self.final_attention:
            final_attn_output = nets['final_attn_layers'](query=attn_output, value=goal_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=goal_gripper_pcd_rel_pos_embedding,
            )[-1]
            final_attn_output = nets['final_slef_attn_layers'](query=final_attn_output, value=final_attn_output,
                query_pos=gripper_pcd_rel_pos_embedding, value_pos=gripper_pcd_rel_pos_embedding,
            )[-1]
            obs_features = einops.rearrange(
                final_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)
                
            obs_features = torch.cat([obs_features, state_feat], dim=-1)     
        else:
            goal_features = einops.rearrange(
                goal_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1)

            obs_features = torch.cat([obs_features, goal_features], dim=-1)    
            
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels
    

    def get_pcd_features(self):
        return self._point_cloud, self._pcd_features

        