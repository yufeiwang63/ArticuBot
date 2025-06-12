import os
import pickle
import numpy as np
import copy
from tqdm import tqdm
import argparse

def mean_center_dataset(zarr_folder_path, data_folder_list, output_zarr_folder_path, keys_to_center = ['state', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd'], enable_datagen_test=True, precision_for_check=1e-4):
    zarr_path_list = []
    zarr_output_path_list = []
    if data_folder_list == None:
        data_folder_list = os.listdir(zarr_folder_path)
    for data_folder in data_folder_list:
        if data_folder == '' or data_folder == ' ':
            continue
        zarr_path_list.append(zarr_folder_path + data_folder)
        zarr_output_path_list.append(output_zarr_folder_path + data_folder)
    for zarr_path, output_zarr_path in tqdm(zip(zarr_path_list, zarr_output_path_list), total=len(zarr_path_list)):
        all_subfolder = os.listdir(zarr_path)
        all_subfolder = sorted(all_subfolder)
        n_episodes = len(all_subfolder)
        zarr_paths = []
        all_paths = []
        output_zarr_paths = []
        output_all_paths = []
        for subfolder in all_subfolder:
            if len(os.listdir(os.path.join(zarr_path, subfolder))) > 10:
                zarr_paths.append(os.path.join(zarr_path, subfolder))
                output_zarr_paths.append(os.path.join(output_zarr_path, subfolder))
        all_paths += zarr_paths
        output_all_paths += output_zarr_paths
        for zarr_path_inner, output_zarr_path_inner  in tqdm(zip(all_paths, output_all_paths), total=len(all_paths)):
            all_substeps = os.listdir(zarr_path_inner)
            all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
            for i, substep in enumerate(all_substeps):
                substep_path = os.path.join(zarr_path_inner, substep)
                os.makedirs(output_zarr_path_inner, exist_ok=True)
                output_substep_path = os.path.join(output_zarr_path_inner, substep)
                data = pickle.load(open(substep_path, 'rb'))
                mean_point_cloud = np.mean(data["point_cloud"], axis=1, keepdims=True)
                for key in keys_to_center:
                    if key != "state":
                        data[key] = data[key] - mean_point_cloud
                    else:
                        data[key][:,0] = data[key][:,0] - mean_point_cloud[0,0,0]
                        data[key][:,1] = data[key][:,1] - mean_point_cloud[0,0,1]
                        data[key][:,2] = data[key][:,2] - mean_point_cloud[0,0,2]
                with open(output_substep_path, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if enable_datagen_test:
                    data = pickle.load(open(substep_path, 'rb'))
                    data_centered = pickle.load(open(output_substep_path, 'rb'))
                    a = np.max(data["point_cloud"] - data_centered["point_cloud"], axis=1)
                    b = np.min(data["point_cloud"] - data_centered["point_cloud"], axis=1)
                    c = mean_point_cloud[0]
                    tol_array = [precision_for_check, precision_for_check, precision_for_check]
                    assert compare_arrays_with_tolerance(a, b, tol_array) and compare_arrays_with_tolerance(b, c, tol_array), f"There maybe issue in data generation. The generated and original point clouds should have the difference eqactly equal to the mean of the point cloud.The values are {a}, {b} and {c}"

def compare_arrays_with_tolerance(arr1, arr2, tol):
        return np.all(np.abs(arr1 - arr2) <= tol)

def list_of_strings(arg):
        return arg.split(',')
def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('-zf','--zarr_folder_path', type=str,default='/scratch/chialiang/dp3_demo/', help='Path to input folder')
    parser.add_argument('-dfl','--data_folder_list', type=list_of_strings, default=None, help='List of folders to center')
    parser.add_argument('-ozf','--output_zarr_folder_path', type=str,default='/project_data/held/pratik/centered_dataset/', help='Folder to store the centered output')
    args = parser.parse_args()

    mean_center_dataset(args.zarr_folder_path, args.data_folder_list, args.output_zarr_folder_path)

if __name__ == "__main__":
    main()