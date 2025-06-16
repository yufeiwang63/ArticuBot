cd weighted_displacement_model
torchrun --standalone --nproc_per_node=1 train_ddp_weighted_displacement.py --batch_size 50 \
    --num_epochs 60  \
    --exp_path weighted_displacement_model/exps \
    --num_train_objects 50 \
    --dataset_prefix data/dp3_demo \
    --exp_name _test_cleaned_code

### num_train_objects can be 10, 50, 100, 200, 300 etc. For all options see `get_dataset_from_pickle` in "weighted_displacement_model/dataset_from_disk.py".