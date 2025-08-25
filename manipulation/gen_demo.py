import argparse
import os
import yaml
import json
import copy
import numpy as np
import pybullet as p
import time
import datetime
import multiprocessing as mp
from termcolor import cprint
from manipulation.utils import build_up_env, save_numpy_as_gif,  parse_center

def get_all_test_configs(root_dir='data/temp', extract_name=None):
    all_tasks = os.listdir(root_dir)
    all_tasks = sorted(all_tasks)
    yaml_configs = []
    solution_paths = []
    obj_ids = []
    if extract_name is not None:
        all_tasks = [x for x in all_tasks if extract_name in x]
        
    for task in all_tasks:
        path = os.path.join(root_dir, task)
        config_path = os.path.join(path, 'base_config.yaml')
        if not os.path.exists(config_path):
            print(f"Config file {config_path} does not exist.")
            continue
        config = yaml.safe_load(open(config_path, "r"))
        solution_path = [x['solution_path'] for x in config if 'solution_path' in x][0]
        obj_ids.append([x['reward_asset_path'] for x in config if 'reward_asset_path' in x][0])
            
        yaml_configs.append(config_path)
        solution_paths.append(solution_path)
        
    return yaml_configs, solution_paths, obj_ids

def get_current_num_demos(path):
    num = 0
    if os.path.exists(path):
        all_experiments = os.listdir(path)
        all_experiments = sorted(all_experiments)
        for exp in all_experiments:
            state_path = os.path.join(path, exp, "states")
            if os.path.exists(state_path):
                if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(path, exp, "all.gif")):
                    num += 1
    return num

def extract_handle_mesh(obj_id):
    from manipulation.extract_handle_mesh import render
    cur_shape_dir = "data/dataset/{}".format(obj_id)
    cur_result_json = os.path.join(cur_shape_dir, 'result.json')
    with open(cur_result_json, 'r') as fin:
        tree_hier = yaml.safe_load(fin)[0]
    data = tree_hier
    render(data, cur_shape_dir)

def create_config_variant(config_path):
    config_variant_paths = os.path.join("/".join(config_path.split("/")[:-1]), "configs")
    if not os.path.exists(config_variant_paths):
        os.makedirs(config_variant_paths)

    # create config variant
    new_config_path = os.path.join(config_variant_paths, f"config_larger_randomization_{try_times+500}.yaml")
        
    base_config = yaml.safe_load(open(config_path, "r"))
    new_config = copy.deepcopy(base_config)

    for config_dict in new_config:
        if 'size' in config_dict:
            size_ratio = np.random.uniform(0.7, 0.9) ### default is 1.5
            config_dict['size'] = size_ratio * config_dict['size']
            config_dict['is_crop_size'] = True

    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, indent=4)

    return new_config_path

def _gen_init_state(q, config_path, env_name, render, far_distance=0.7, near_distance=0.3):
    # get table bbox
    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        if 'center' in config_dict:
            base_pos = parse_center(config_dict['center'])
            print("base pos: ", base_pos)

    # create env
    env, _ =  build_up_env(config_path, env_name, render=render)
    env.reset()

    # get the joint limits for the robot arm, using a smaller range
    initial_joint_angles = [0 for _ in range(7)]
    low = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9]
    high = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]
    for i in range(7):
        joint_range = high[i] - low[i]
        low[i] += joint_range * 0.2
        high[i] -= joint_range * 0.2    

    # get the initial position and orientation of the object
    object_id = env.urdf_ids[object_name]   
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    init_euler = p.getEulerFromQuaternion(init_orient)

    good_init_pos = False
    while not good_init_pos:
        height = 0

        # randomly sample the position and orientation of the object
        new_pos = np.array([0, 0, init_pos[2]])
        new_pos[0] = np.random.uniform(-0.1, 0.1) + base_pos[0]
        new_pos[1] = np.random.uniform(-0.1, 0.1) + base_pos[1]
        new_pos[2] = height + init_pos[2]

        init_angle = None
        info = env._get_info()
        handle_joint_id = env.handle_joint
        print("handle joint id: ", handle_joint_id)
        joint_limit_max = p.getJointInfo(env.urdf_ids[object_name], handle_joint_id, physicsClientId=env.id)[9]
        joint_limit_min = p.getJointInfo(env.urdf_ids[object_name], handle_joint_id, physicsClientId=env.id)[8]
        joint_limit = joint_limit_max - joint_limit_min
        init_angle = np.random.uniform(0, 0.2) * joint_limit

        new_orient = p.getQuaternionFromEuler([init_euler[0], init_euler[1], base_euler[2] + np.random.uniform(-np.pi / 6, np.pi / 6)])

        print("new pos: ", new_pos)
        print("new orient: ", p.getEulerFromQuaternion(new_orient))
        p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)

        # ensure the handle is not too far away from the robot base
        info = env._get_info()
        handle_joint_id = env.handle_joint
        handle_pos = info['handle_pos']
        robot_base = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]
        distance_base_to_handle = np.linalg.norm(handle_pos - np.array(robot_base))

        # randomly set the joint angle of the handle
        joint_limit_low, joint_limit_high = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)[8:10]
        max_opened_joint = joint_limit_low + 0.2 * (joint_limit_high - joint_limit_low)
        random_joint = np.random.uniform(joint_limit_low, max_opened_joint)
        p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)

        # add a few steps to let the robot arm settle down
        for _ in range(5):
            p.stepSimulation()

        joint_state = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
        if abs(joint_state - random_joint) > 1e-3:
            continue

        for test_time in range(100):
            # randomly sample the joint angles of the robot arm
            for i in range(7):
                initial_joint_angles[i] = np.random.uniform(low[i], high[i])   
            env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
            
            p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)

            # add a few steps to let the robot arm settle down
            for _ in range(5):
                p.stepSimulation()
            
            joint_state = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
            if abs(joint_state - random_joint) > 1e-3:
                continue

            # check if the robot arm is in contact with the object
            contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
            closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
            if len(contact_points) > 0 or len(closest_points) > 0:
                continue

            # check if the robot arm is in contact with the plane
            link_contact = False
            num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
            for link_idx in range(1, num_links):
                contact_points = p.getClosestPoints(bodyA=env.robot.body, linkIndexA=link_idx, bodyB=env.urdf_ids['plane'], distance=0.01, physicsClientId=env.id)
                if len(contact_points) > 0:
                    link_contact = True
                    break
            if link_contact: 
                continue
            
            # check if the robot end effector is in a good position to grasp the handle
            robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
            distance = np.linalg.norm(handle_pos - robot_eef_pos)
            if distance < far_distance and distance > near_distance:
                good_init_pos = True
                new_pos = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)[0]
                break
        
    if not good_init_pos:
        # fail to find a good initial position
        q.put(False)
        return False

    # found a good initial position
    # randomly sample the initial finger angle of the robot gripper
    initial_finger_angle = np.random.uniform(env.robot.finger_fully_close_joint_angle, env.robot.finger_fully_open_joint_angle)
    
    # add a few steps to let the robot gripper settle down
    env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [initial_finger_angle, initial_finger_angle], set_instantly=True)
    p.stepSimulation()

    # save the initial state
    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'center' in config_dict:
            new_pos = [new_pos[0], new_pos[1], height]
            config_dict['center'] = str(tuple(new_pos))
            config_dict['orientation'] = str(tuple(new_orient))
            config_dict['init_angle'] = init_angle
            config_dict['is_crop_size'] = False
            if 'initial_joint_angles' not in config_dict:
                config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
            if 'initial_finger_angle' not in config_dict:
                config_dict['initial_finger_angle'] = initial_finger_angle
        if "set_joint_angle_object_name" in config_dict:
            config_dict['set_joint_angle_object_name'] = object_name
            config_dict['set_joint_angle_joint_id'] = handle_joint_id
            config_dict['set_joint_angle_joint_angle'] = random_joint
        if 'initial_joint_angles' in config_dict:
            config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
        if 'initial_finger_angle' in config_dict:
            config_dict['initial_finger_angle'] = initial_finger_angle

    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
        
    env.close()
    q.put(True)
    return True

# handle memory leak             
def gen_init_state(config_path, env_name, render, far_distance=0.7, near_distance=0.3, ):
    q = mp.Queue()  
    p = mp.Process(target=_gen_init_state, args=(q, config_path, env_name, render, far_distance, near_distance))
    p.start()
    p.join()
    success = q.get()
    return success

def _execute(q, config_path, env_name, solution_path, experiment_path, time_string=None):
    # get time string
    if time_string is None:
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    if not os.path.exists(solution_path):
        os.makedirs(solution_path)

    experiment_path = os.path.join(experiment_path, time_string)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    os.system("cp {} {}".format(config_path, os.path.join(experiment_path, "task_config.yaml")))

    config = yaml.safe_load(open(config_path, "r"))
    
    # execute primitive
    print("execute primitive")
    env, _ = build_up_env(config_path, env_name)
    env.primitive_save_path = experiment_path
    np.random.seed(time.time_ns() % 2**32)

    env.reset()
    rgbs, states = env.execute()
    p.disconnect(env.id)
    
    # execute the primitive from the environment to get the trajectory.
    if len(states) > 10:
        with open(os.path.join(experiment_path, "last_state_files.txt"), 'w') as f:
            f.write("\n".join(str(states[-1])))
        save_numpy_as_gif(np.array(rgbs), "{}/{}.gif".format(experiment_path, "all"))
        q.put(True)
        return True
    else:
        q.put(False)
        return False
    
# handle memory leak
def execute(config_path, env_name, solution_path, experiment_path):
    q = mp.Queue()  
    p = mp.Process(target=_execute, args=(q, config_path, env_name, solution_path, experiment_path, None))
    p.start()
    p.join()
    success = q.get()
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demos.")
    parser.add_argument("--exp_name", type=str, default="debug") # experiment name to save this demonstration, e.g., 0824-test-demo-gen
    parser.add_argument("--env_name", type=str, default="articulated")
    parser.add_argument("--extract_name", type=str, default=None) # which object to generate the demo, e.g., 41510
    parser.add_argument("--num_to_generate", type=int, default=100)
    parser.add_argument("--max_try_times", type=int, default=500)
    parser.add_argument("--root_dir", type=str, default=None) # folder name: data/diverse_objects_rest/
    parser.add_argument("--robot_name", type=str, default="panda") 
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--log_path", type=str, default="./log.txt") 


    args = parser.parse_args()
    root_dir = args.root_dir
    extract_name = args.extract_name
    exp_name = args.exp_name
    log_path = args.log_path
    env_name = args.env_name

    near_distance = 0.15
    far_distance = 0.5

    generate_times = 1 

    path = "{}/{}/experiment/{}".format(root_dir, extract_name, exp_name)
    all_config_paths, all_solution_paths, all_obj_ids = get_all_test_configs(root_dir=root_dir, extract_name=extract_name)
    num_demos = get_current_num_demos(path)

    for i in range(generate_times):
        done = False
        try_times = 0
        while True:
            for config_path, solution_path, obj_id in zip(all_config_paths, all_solution_paths, all_obj_ids):
                if try_times >= 20 and num_demos == 0:
                    done = True
                    # too many tries, fail to generate demo
                    break
                        
                if try_times >= args.max_try_times:
                    done = True
                    # too many tries, skip 
                    break
                            
                if num_demos >= args.num_to_generate:
                    print("Generated enough demos: ", num_demos)
                    done = True
                    # generated enough demos
                    break
                
                extract_handle_mesh(obj_id)
                # create experiment folder, meta info and config variant

                experiment_path = os.path.join(solution_path, "experiment", exp_name)
                if not os.path.exists(experiment_path):
                    os.makedirs(experiment_path)
                
                meta_info_path = os.path.join(experiment_path, "meta_info.json")
                with open(meta_info_path, 'w') as f:
                    json.dump(args.__dict__, f, indent=4)

                new_config_path = create_config_variant(config_path)

                # generate initial state
                success = gen_init_state(new_config_path, env_name, args.render, near_distance=near_distance, far_distance=far_distance)

                if not success:
                    continue

                sucess = execute(new_config_path, env_name, solution_path, experiment_path)
                try_times += 1
                if sucess:
                    num_demos += 1
                    cprint("generated demo: {}".format(num_demos), color="cyan")
                
            if done:
                break
