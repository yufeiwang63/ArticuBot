
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

pointcloud_num=4500
encoding_mode="keep_position_feature_in_attention_feature"
horizon=8
n_obs_steps=2 # 2 or 4
training_epoches=101
train_ratio=0.9 # for generalization
num_load_episodes=1000    # for generalization
pc_channel=3 # we should modify this
batch_size=400 #######
encoder_type=act3d
use_mlp=1
in_channels=3 ####
augmentation_pcd=true

time_stamp=$(date +%m%d%H%M)
exp_name="test-cleaned-low-level-code"

action_dim=10
agent_pos_dim=10

### can be 10_low_level, 100_low_level, etc. See "get_zarry_paths" in 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robogen_dataset.py for all options. 
num_train_objects=50_low_level

torchrun --standalone --nproc_per_node=1 \
    train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}"  \
    task.dataset.zarr_path="${num_train_objects}" \
    task.env_runner.demo_experiment_path="[]" \
    task.env_runner.experiment_name="[]" \
    task.env_runner.experiment_folder="[]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    horizon="${horizon}" \
    n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    policy.encoder_type="${encoder_type}" \
    policy.encoder_output_dim=60 \
    policy.act3d_encoder_cfg.in_channels=${in_channels} \
    policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
    policy.act3d_encoder_cfg.mode="${encoding_mode}" \
    policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
    training.num_epochs="${training_epoches}" \
    training.checkpoint_every=20 \
    task.dataset.train_ratio="${train_ratio}" \
    task.dataset.num_load_episodes=${num_load_episodes} \
    task.dataset.augmentation_pcd="${augmentation_pcd}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    policy.noise_model_type=unet \
    policy.policy_type=low_level



    
