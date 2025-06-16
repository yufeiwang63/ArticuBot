import json
import os
import numpy as np
import argparse

def plot_open_door_results(directory, num_objs=10, start_idx=0):
    json_results = []
    one_exp_results = []
    
    for j in range(start_idx, num_objs):
        json_file = os.path.join(directory, f"opened_joint_angles_{j}.json")
        if not os.path.exists(json_file): continue
        with open(json_file) as f:
            data = json.load(f)
            opened_joint_angles = []
            expert_angles = []
            initial_angles = []
            for key in data:
                if data[key]["expert_door_joint_angle"] <= data[key]['initial_joint_angle']:
                    continue
                opened_joint_angles.append(data[key]["final_door_joint_angle"])
                expert_angles.append(data[key]["expert_door_joint_angle"] if "46462" not in key else 0.27)
                initial_angles.append(data[key]['initial_joint_angle'])

            normalized_performance = (np.array(opened_joint_angles) - np.array(initial_angles)) / (np.array(expert_angles) - np.array(initial_angles))
            normalized_performance[normalized_performance > 1] = 1
            normalized_performance = normalized_performance[~np.isnan(normalized_performance)]
            normalized_performance = normalized_performance[~np.isinf(normalized_performance)]
            if len(normalized_performance) >= 10:
                normalized_performance = np.mean(normalized_performance)
                one_exp_results.append(normalized_performance)
            
    json_results.append(one_exp_results)
    json_results = np.array(json_results)
    mean_result = np.mean(json_results, axis=0)
    std_result = np.std(json_results, axis=0)
    
    return mean_result, std_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--num_objs", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()
    mean_result, std_result = plot_open_door_results(args.d, args.num_objs, args.start_idx)
    print("============== normalized performance ==================")
    mean_result = np.round(mean_result, 3)
    for x in mean_result:
        print(x, end=',')
    print("")
    print("mean: ", np.mean(mean_result))
    
    
