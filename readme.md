
<!-- <div align="center">
  <img width="500px" src="imgs/logo.png"/> -->
  
  <div align="center">

  # ArticuBot: Learning Universal Articulated Object Manipulation Policy via Large Scale Simulation
### RSS 2025

</div>


<!-- ---
<div align="center">
  <img src="imgs/teaser.png"/>
</div>  -->

<p align="center">
    <a href='https://arxiv.org/abs/2503.03045'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://articubot.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>
This is the official repository of the paper:

**[ArticuBot: Learning Universal Articulated Object Manipulation Policy via Large Scale Simulation](https://articubot.github.io/)**, RSS 2025  
[Yufei Wang*](https://yufeiwang63.github.io/), [Ziyu Wang*](https://articubot.github.io/), [Mino Nakura&dagger;](https://articubot.github.io/), [Pratik Bhowal&dagger;](https://articubot.github.io/), [ Chia-Liang Kuo&dagger;](https://sites.google.com/view/chialiangkuo), [Yi-Ting Chen](https://sites.google.com/site/yitingchen0524/home), [Zackory Erickson&Dagger;](https://zackory.com/), [David Held&Dagger;](https://davheld.github.io/)   
(*&dagger; equal contribution, &Dagger; equal advising)

ArticuBot learns a universal policy for manipulating diverse articulated objects. It first generates a large amount of data in simulation, and then distills them into a visual policy via hierarchical imitation learning. Finally, the learned policy can be zero-shot transferred to the real world. 
<p align="center">
  <img src="data/articubot.gif" alt="Demo GIF" width="500">
</p>

## Installation
Clone this git repo.
We recommend working with a conda environment.

- First create the articubot conda environment:
```
conda env create -f environment.yaml
conda activate articubot
```

- Then install some additional dependencies for 3d diffusion policy (which our low-level goal conditioned policy builds on):
```
cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy
pip install -e . 
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
```

- Install pybullet:
```
cd bullet3
pip install -e .
```

- Install fpsample for fasthest point sampling:
```
pip install fpsample
```
The installtaion might give an error which requires you to install Rust first. Following the prompt in the installtaion error message should be sufficient. 
If the installation runs into any other issue, check https://github.com/leonardodalinky/fpsample.

- (Optional) In addition to pybullet's default IK solver, we use tracIK as an additional ik solver which might be more accurate and give better IK solutions. If you wish to install tracIK, follow instructions here: https://github.com/mjd3/tracikpy. 

The above should be sufficient for training and evaluating articubot policies. 


## Training and evaluation dataset

All datasets we use can be downloaded from this [google drive link](https://drive.google.com/drive/folders/1lbpoo8SqNuLWTjMyvO5RWnBd0XpGq6C4?usp=sharing).

- We use [PartNet-Mobility](https://sapien.ucsd.edu/browse) as our simulation assets. We provide a parsed version in the above google drive link, named `dataset.zip`. After downloading, unzip it to `data/dataset`.
- The high-level policy training dataset is named `dp3_demo.zip`. Unzip it to `data/dp3_demo`. 
- The high-level policy training dataset is named `dp3_demo_combined_2_step_0.zip`. Unzip it to `data/dp3_demo_combined_2_step_0`.  
- Some files will also be needed for running evaluation (the simulator init states). 
They are named `diverse_objects.zip`. Unzip it to `data/diverse_objects`.

You can run `python scripts/visualization_scripts/check_data.py` to visualize the stored datasets. 

## High and low-level policy training
Assume you have downloaded the training dataset following the above instructions.

For training the high-level policy: 
```
source prepare.sh
bash scripts/weighted-displacement-high-level/train-weighted-displacement.sh
```
Change `num_train_objects` in this script for training with different number of objects. 
The training logs for the low-level policy will be at `3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/`


For training the low-levle policy:
```
source prepare.sh
bash scripts/low-level/train_unet_diffusion_low_level.sh
```
Change `num_train_objects` in the script for training with different number of objects. 
The training logs for the high-level policy will be at `weighted_displacement_model/exps/`


## Evaluate trained policies
Assume you have downloaded the needed evaluation files following the above instructions. 

We provide pretrained high-level and low-level ckpts through this [google drive link](https://drive.google.com/drive/folders/1lbpoo8SqNuLWTjMyvO5RWnBd0XpGq6C4?usp=sharing):

The low-level policy ckpt is named `low-level-ckpt.zip`. Unzip it to `data/low-level-ckpt`.  
The high-level policy ckpt is named `high_level_300_obj_ckpt.pth`. Download and put it to `data/high_level_300_obj_ckpt.pth`.

To evaluate a trained high and low-level policy:
```
source prepare.sh
bash scripts/weighted-displacement-high-level/eval-weighted-displacement-high-level.sh
```
The evaluation results will be saved at `3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/`.

To print the quantitive numbers, you can run `python scripts/print_eval_results.py --d your_eval_run_results_dir`

## Demonstration generation
We are currently in the process of cleaning the data generation code, so stay tuned!

### Open Motion Planning Library (OMPL) for demonstration generation
OMPL is only needed for generating the training demonstrations. It is not needed for training the policy.

If your system is ubuntu 20.04 or higher, use the prebuilt wheels for python 3.9: https://drive.google.com/file/d/1dGiq8_CqIPFWqjqyXJzT7yp2z0PVLEdX/view?usp=sharing (See https://github.com/ompl/ompl/releases/tag/prerelease for more wheels for different python versions). Run `pip install your_downloaded_ompl_wheel` to install the library (do this in the articubot conda env). 

If your system is ubuntu 18.04, to install OMPL, run
```
cd scripts
./install_ompl_1.5.2.sh --python
```
which will install the ompl with system-wide python. Note at line 19 of the installation script OMPL requries you to run `sudo apt-get -y upgrade`. A successful installation will take 5-6 hours. If not, that means your installtaion is not complete. If the installation is not complete, try installing by running the commands in install_ompl_1.5.2.sh line by line. 
Then, export the installation to the conda environment to be used with ArticuBot:
```
echo "path_to_your_ompl_installation_from_last_step/OMPL/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/articubot/lib/python3.9/site-packages/ompl.pth
```
remember to change the path to be your ompl installed path and conda environment path.


## Acknowledgement
We thank the authors of the following repositories for open-sourcing their code, which our codebase is built upon:
- DP3: https://github.com/YanjieZe/3D-Diffusion-Policy
- RoboGen: https://github.com/Genesis-Embodied-AI/RoboGen
- Act3D: https://github.com/zhouxian/act3d-chained-diffuser
- PointNet++: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- OMPL: https://ompl.kavrakilab.org/
- Bullet: https://github.com/bulletphysics/bullet3

## Citation
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{Wang2025articubot,
  title={ArticuBot: Learning Universal Articulated Object Manipulation Policy via Large Scale Simulation},
  author={Wang, Yufei and Wang, Ziyu and Nakura, Mino and Bhowal, Pratik and Kuo, Chia-Liang and Chen, Yi-Ting and Erickson, Zackory and Held, David},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}   
```
