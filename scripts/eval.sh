### evaluation without any camera randomization
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy
python eval_robogen.py  --low_level_exp_dir "${PROJECT_DIR}/data/low-level-ckpt" \
  --low_level_ckpt_name low-level.ckpt \
  --high_level_ckpt_name "${PROJECT_DIR}/data/high_level_300_obj_ckpt.pth"   \
  --eval_exp_name test_cleaned_code
#   --real_world_camera 1 \  ### if you want to evaluate with a camera distribution closer to our real-world setting (not seen during training)
#   --noise_real_world_pcd 1 \  ### if you want to evaluate with noise added to the depth image to approximate the real-world depth camera noises (not used during training)
#   --randomize_camera 1 \ ### if you want to randomize the camera pose (seen during training)