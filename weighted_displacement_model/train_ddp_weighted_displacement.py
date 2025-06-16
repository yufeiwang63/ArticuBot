from tqdm import tqdm
import argparse
import einops
import wandb
import datetime
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from weighted_displacement_model.dataset_from_disk import get_dataset_from_pickle
from weighted_displacement_model.model_invariant import PointNet2_super

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)

    input_channel = 5 if args.add_one_hot_encoding else 3
    output_dim = 13 
    model = PointNet2_super(num_classes=output_dim, input_channel=input_channel).to(device)
    
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Successfully load model from: ", args.load_model_path)
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    output_dir = str(datetime.date.today())
    if args.use_all_data: output_dir = output_dir + "_use_all_data" 
    output_dir = output_dir + "_" + str(args.num_train_objects) + "-obj"
    if args.add_one_hot_encoding: output_dir = output_dir + "_one_hot"
    output_dir += args.exp_name
    if not args.exp_path.startswith('/'):
        args.exp_path = os.path.join(os.environ['PROJECT_DIR'], args.exp_path)
    args.exp_path = os.path.join(args.exp_path, output_dir)

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = DDP(model, device_ids=[gpu_id])

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        wandb_run = wandb.init(
                project="articubot-high-level-weighted-displacement",
                name=str(output_dir),
                dir=str(args.exp_path),
            )
        wandb.config.update(
            {
                "output_dir": args.exp_path,
                "lr": args.lr,
                "weight_loss_weight": args.weight_loss_weight,
                "batch_size": args.batch_size
            }
        )
        
        config_dict = args.__dict__
        wandb.config.update(config_dict)

        # save the config file
        with open(os.path.join(args.exp_path, "config.txt"), "w") as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

    print("trying to load dataset")
    dataset = get_dataset_from_pickle(beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, 
                                      use_all_data=args.use_all_data, dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, 
                shuffle=False,
                sampler=sampler,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
            )

    global_step = 0
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        accumulated_displacement_loss = 0.0
        accumulated_weighting_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            pointcloud, gripper_pcd, goal_gripper_pcd = data
            # pointcloud: B, N=4500, 3
            # gripper_pcd: B, 4, 3
            # goal_gripper_points: B, 4, 3
            gripper_points = goal_gripper_pcd
            
            if args.add_one_hot_encoding:
                # for pointcloud, we add (1, 0)
                # for gripper_pcd, we add (0, 1)
                pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
                pointcloud_one_hot[:, :, 0] = 1
                pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2)
                gripper_pcd_one_hot[:, :, 1] = 1
                gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
            else:
                inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3

            # calculate the displacement from every point to the goal gripper to get the labels with shape B, N, 4, 3
            labels = gripper_points.unsqueeze(1) - inputs[:, :, :3].unsqueeze(2)
            B, N, _, _ = labels.shape
            labels = labels.view(B, N, -1) # B, N, 12

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            
            outputs = model(inputs) # B, N, 13
            weights = outputs[:, :, -1] # B, N
            outputs = outputs[:, :, :-1] # B, N, 12
            loss = criterion(outputs, labels)
            accumulated_displacement_loss += loss.item()

            inputs = inputs.permute(0, 2, 1)
            outputs = outputs.view(B, N, 4, 3) # displacement from each point to the goal gripper, B, N, 4, 3
            outputs = outputs + inputs[:, :, :3].unsqueeze(2) # B, N, 4, 3

            # softmax the weights
            weights = torch.nn.functional.softmax(weights, dim=1)
            
            # sum the displacement of the predicted gripper point cloud according to the weights
            outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
            outputs = outputs.sum(dim=1)
            avg_loss = criterion(outputs, gripper_points.to(device))

            loss = loss + avg_loss * args.weight_loss_weight
            accumulated_weighting_loss += (avg_loss * args.weight_loss_weight).item()

            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()

            log_interval = len(dataloader) // 10
            if (i+1) % log_interval == 0 and os.environ['LOCAL_RANK'] == '0':
                print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / log_interval}")
                
                log_info = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "total_loss": running_loss / log_interval,
                    "displacement_loss": accumulated_displacement_loss / log_interval,
                    "weighting_loss": accumulated_weighting_loss / log_interval,
                }

                wandb_run.log(log_info, step=global_step)

                running_loss = 0.0
                accumulated_displacement_loss = 0.0
                accumulated_weighting_loss = 0.0

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 and os.environ['LOCAL_RANK'] == '0':
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.module.state_dict(), save_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_objects', default=200)
    parser.add_argument('--dataset_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/weighted_displacement_model/exps")
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--weight_loss_weight', type=float, default=10)
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    train(args)
    destroy_process_group()