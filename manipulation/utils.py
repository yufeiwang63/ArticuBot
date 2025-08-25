import os
import yaml
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import os.path as osp
import pybullet as p
import os
import json
import multiprocessing
from multiprocessing import Process
import multiprocessing.pool
import pickle
import copy
import importlib
from PIL import Image, ImageSequence
import multiprocessing
import objaverse
import trimesh
import random
import scipy
from objaverse_utils.utils import text_to_uid_dict, partnet_mobility_dict, sapaien_cannot_vhacd_part_dict
from typing import Optional, List, Dict, Any, Tuple
from scipy import ndimage

# Chialiang
from scipy.spatial.transform import Rotation as R

workspace = os.environ['PWD']
data_dir = os.path.join(workspace, "data")

default_config = {
    "gui": False,
}

def normalize_obj(obj_file_path):
    vertices = []
    with open(osp.join(obj_file_path), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
    
    vertices = np.array(vertices).reshape(-1, 3)
    vertices = vertices - np.mean(vertices, axis=0) # center to zero
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalize to -1, 1

    with open(osp.join(obj_file_path.replace(".obj", "_normalized.obj")), 'w') as f:
        vertex_idx = 0
        for line in lines:
            if line.startswith("v "):
                line = "v " + " ".join([str(x) for x in vertices[vertex_idx]]) + "\n"
                vertex_idx += 1
            f.write(line)

def down_load_single_object(name, uids=None, candidate_num=5, vhacd=True, debug=False, task_name=None, task_description=None):
    if uids is None:
        if name in text_to_uid_dict:
            uids = text_to_uid_dict[name]
        else:
            from objaverse_utils.find_uid_utils import find_uid
            uids = find_uid(name, candidate_num=candidate_num, debug=debug, task_name=task_name, task_description=task_description)
            if uids is None:
                return False

    processes = multiprocessing.cpu_count()
   
    for uid in uids:
        save_path = osp.join(os.environ["PROJECT_DIR"], "objaverse_utils/data/obj", "{}".format(uid))
        if not osp.exists(save_path):
            os.makedirs(save_path)
        if osp.exists(save_path + "/material.urdf"):
            continue

        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=processes
        )
        
        test_obj = (objects[uid])
        scene = trimesh.load(test_obj)

        try:
            trimesh.exchange.export.export_mesh(
                scene, osp.join(save_path, "material.obj")
            )
        except:
            if debug:
                return False
            # print("cannot export obj for uid: ", uid)
            uids.remove(uid)
            if name in text_to_uid_dict and uid in text_to_uid_dict[name]:
                text_to_uid_dict[name].remove(uid)
            continue

        # we need to further parse the obj to normalize the size to be within -1, 1
        if not osp.exists(osp.join(save_path, "material_normalized.obj")):
            normalize_obj(osp.join(save_path, "material.obj"))

        # we also need to parse the obj to vhacd
        if vhacd:
            if not osp.exists(osp.join(save_path, "material_normalized_vhacd.obj")):
                run_vhacd(save_path)

        # for pybullet, we have to additionally parse it to urdf
        obj_to_urdf(save_path, scale=1, vhacd=vhacd) 

    return True

def download_and_parse_objavarse_obj_from_yaml_config(config_path, candidate_num=10, vhacd=True):

    config = None
    while config is None:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

    task_name = None
    task_description = None
    for obj in config:
        if 'task_name' in obj.keys():
            task_name = obj['task_name']
            task_description = obj['task_description']
            break

    for obj in config:
        if 'type' in obj.keys() and obj['type'] == 'mesh' and 'uid' not in obj.keys():
            print("{} trying to download object: {} {}".format("=" * 20, obj['lang'], "=" * 20))
            success = down_load_single_object(obj["lang"], candidate_num=candidate_num, vhacd=vhacd, 
                                              task_name=task_name, task_description=task_description)
            if not success:
                print("failed to find suitable object to download {} quit building this task".format(obj["lang"]))
                return False
            obj['uid'] = text_to_uid_dict[obj["lang"]]
            obj['all_uid'] = text_to_uid_dict[obj["lang"] + "_all"]

            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=4)

    return True

def load_gif(gif_path):
    img = Image.open(gif_path)
    # Extract each frame from the GIF and convert to RGB
    frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(img)]
    # Convert each frame to a numpy array
    frames_arrays = [np.array(frame) for frame in frames]
    return frames_arrays

def build_up_env(task_config=None, env_name=None, task_name=None, restore_state_file=None, return_env_class=False, 
                    action_space='delta-translation', render=False, randomize=False, 
                    obj_id=0, random_object_translation: Optional[List]=None, **kwargs,
                ):
    config = yaml.safe_load(open(task_config, "r"))
    link_name = 'link_0'
    init_angle = None
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
        if 'init_angle' in config_dict:
            init_angle = config_dict['init_angle']

    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = task_name
    save_config['object_name'] = object_name
    save_config['link_name'] = link_name
    save_config['init_angle'] = init_angle
    save_config['restore_state_file'] = restore_state_file
    save_config['translation_mode'] = action_space
    save_config['gui'] = render
    save_config['randomize'] = randomize
    save_config['obj_id'] = obj_id
    save_config['random_object_translation'] = random_object_translation
    for key, value in kwargs.items():
        save_config[key] = value

    ### you might want to restore to a specific state
    # module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    # env_class = getattr(module, task_name)
    module = importlib.import_module("manipulation.envs.{}".format(env_name))
    env_class = getattr(module, env_name)
    env = env_class(**save_config)


    if not return_env_class:
        return env, save_config
    else:
        return env, save_config, env_class

class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc

def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

def obj_to_urdf(obj_file_path, scale=1, vhacd=True, normalized=True, obj_name='material'):
    header = """<?xml version="1.0" ?>
<robot name="cube.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0.0 0.02 0.0"/>
       <mass value=".1"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
"""

    all_files = os.listdir(obj_file_path)
    png_file = None
    for x in all_files:
        if x.endswith(".png"):
            png_file = x
            break

    if png_file is not None:
        material = """
         <material name="texture">
        <texture filename="{}"/>
      </material>""".format(osp.join(obj_file_path, png_file))        
    else:
        material = """
        <material name="yellow">
            <color rgba="1 1 0.4 1"/>
        </material>
        """

    obj_file = "{}.obj".format(obj_name) if not normalized else "{}_normalized.obj".format(obj_name)
    visual = """
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
      {}
    </visual>
    """.format(osp.join(obj_file_path, obj_file), scale, scale, scale, material)

    if normalized:
        collision_file = '{}_normalized_vhacd.obj'.format(obj_name) if vhacd else "{}_normalized.obj".format(obj_name)
    else:
        collision_file = '{}_vhacd.obj'.format(obj_name) if vhacd else "{}.obj".format(obj_name)
    collision = """
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
             <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
    </collision>
  </link>
  </robot>
  """.format(osp.join(obj_file_path, collision_file), scale, scale, scale)
    


    urdf =  "".join([header, visual, collision])
    with open(osp.join(obj_file_path, "{}.urdf".format(obj_name)), 'w') as f:
        f.write(urdf)

def run_vhacd(input_obj_file_path, normalized=True, obj_name="material"):
    p.connect(p.DIRECT)
    if normalized:
        name_in = os.path.join(input_obj_file_path, "{}_normalized.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_normalized_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    else:
        name_in = os.path.join(input_obj_file_path, "{}.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    p.vhacd(name_in, name_out, name_log)

def parse_center(center):   
    if center.startswith("(") or center.startswith("["):
        center = center[1:-1]

    center = center.split(",")
    center = [float(x) for x in center]
    return np.array(center)

def run_vhacd_with_timeout(args):
    name_in, name_out, name_log, urdf_file_path, obj_file_name = args
    id = p.connect(p.DIRECT)
    proc = Process(target=p.vhacd, args=(name_in, name_out, name_log))

    proc.start()

    # Wait for 10 seconds or until process finishes
    proc.join(200)

    # If thread is still active
    if proc.is_alive():
        print("running too long... let's kill it...")

        # Terminate
        proc.kill()
        proc.join()

        if urdf_file_path not in sapaien_cannot_vhacd_part_dict.keys():
            sapaien_cannot_vhacd_part_dict[urdf_file_path] = []
        sapaien_cannot_vhacd_part_dict[urdf_file_path].append(obj_file_name)

        p.disconnect(id)
        return False

    else:
        print("process finished")
        p.disconnect(id)
        return True

   

def preprocess_urdf(urdf_file_path, num_processes=6):
    new_lines = []
    with open(urdf_file_path, 'r') as f:
        lines = f.readlines()
        
    num_lines = len(lines)
    l_idx = 0
    to_process_args = []
    while l_idx < num_lines:
        line_1 = lines[l_idx]

        if "<collision>" in line_1:
            new_lines.append(line_1)

            for l_idx_2 in range(l_idx + 1, num_lines):
                line_2 = lines[l_idx_2]

                if ".obj" in line_2:
                    start_idx = line_2.find('filename="') + len('filename="')
                    end_idx = line_2.find('.obj') + len('.obj')
                    obj_file_name = line_2[start_idx:end_idx]
                    obj_file_path = osp.join(osp.dirname(urdf_file_path), obj_file_name)
                    # import pdb; pdb.set_trace()
                    name_in = obj_file_path
                    name_out = obj_file_path[:-4] + "_vhacd.obj"
                    name_log = obj_file_path[:-4] + "_log.txt"

                    if not osp.exists(name_out) and obj_file_name not in sapaien_cannot_vhacd_part_dict.get(urdf_file_path, []):
                        to_process_args.append([name_in, name_out, name_log, urdf_file_path, obj_file_name])
                        new_lines.append("to_be_processed, {}".format(line_2))
                    else:
                        new_name = line_2.replace(obj_file_name, obj_file_name[:-4] + '_vhacd.obj')
                        new_lines.append(new_name)
                
                elif "</collision>" in line_2:
                    new_lines.append(line_2)
                    l_idx = l_idx_2 
                    break

                else:
                    new_lines.append(line_2)
            
        else:
            new_lines.append(line_1)

        l_idx += 1

    # do vhacd in parallel, each has a timeout of 200 seconds
    with NonDaemonPool(processes=num_processes) as pool: 
        results = pool.map(run_vhacd_with_timeout, to_process_args)

    processed_idx = 0
    for l_idx in range(len(new_lines)):
        if "to_be_processed" in new_lines[l_idx]:
            if results[processed_idx]:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_name = new_name.replace(".obj", "_vhacd.obj")
                new_lines[l_idx] = new_name
            else:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_lines[l_idx] = new_name
            processed_idx += 1

    new_path = urdf_file_path.replace(".urdf", "_vhacd.urdf")    
    with open(new_path, 'w') as f:
        f.writelines("".join(new_lines))

    with open(f"{data_dir}/sapien_cannot_vhacd_part.json", 'w') as f:
        json.dump(sapaien_cannot_vhacd_part_dict, f, indent=4)

    return new_path


def parse_config(config, obj_id=None, use_vhacd=True, default_initial_joint_angle=0):
    urdf_paths = []
    urdf_sizes = []
    urdf_locations = []
    urdf_orientations = []
    urdf_names = []
    urdf_types = []
    urdf_on_tables = []
    urdf_movables = []
    urdf_crop_sizes = []
    use_table = False
    articulated_joint_angles = {}
    spatial_relationships = []
    distractor_config_path = None

    robot_initial_joint_angles = [0.0, 0.0, 0.0, -0.4, 0.0, 0.4, 0.0]
    initial_finger_angle = default_initial_joint_angle

    for obj in config:
        # print(obj)

        if "use_table" in obj.keys():
            use_table = obj['use_table']

        if "set_joint_angle_object_name" in obj.keys():
            new_obj = copy.deepcopy(obj)
            new_obj.pop('set_joint_angle_object_name')
            articulated_joint_angles[obj['set_joint_angle_object_name']] = new_obj

        if "spatial_relationships" in obj.keys():
            spatial_relationships = obj['spatial_relationships']

        if 'task_name' in obj.keys() or 'task_description' in obj.keys():
            continue

        if "distractor_config_path" in obj.keys():
            distractor_config_path = obj['distractor_config_path']

        if 'initial_joint_angles' in obj.keys():
            initial_joint_angles = obj['initial_joint_angles']
            initial_joint_angles = parse_center(initial_joint_angles)
            robot_initial_joint_angles = initial_joint_angles

        if 'initial_finger_angle' in obj.keys():
            initial_finger_angle = obj['initial_finger_angle']
            initial_finger_angle = float(initial_finger_angle)
            

        if "type" not in obj.keys():
            continue
        
        if obj['type'] == 'mesh':
            if 'uid' not in obj.keys():
                continue
            if obj_id is None:
                uid = obj['uid'][np.random.randint(len(obj['uid']))]
            else:
                uid = obj['uid'][obj_id]
                
            urdf_file_path = osp.join("objaverse_utils/data/obj", "{}".format(uid), "material.urdf")
            if not os.path.exists(urdf_file_path):
                down_load_single_object(name=obj['lang'], uids=[uid])
            

            if not use_vhacd:
                new_urdf_file_path = urdf_file_path.replace("material.urdf", "material_non_vhacd.urdf")
                new_urdf_lines = []
                with open(urdf_file_path, 'r') as f:
                    urdf_lines = f.readlines()
                for line in urdf_lines:
                    if 'vhacd' in line:
                        new_line = line.replace("_vhacd", "")
                        new_urdf_lines.append(new_line)
                    else:
                        new_urdf_lines.append(line)
                with open(new_urdf_file_path, 'w') as f:
                    f.writelines(new_urdf_lines)
                urdf_file_path = new_urdf_file_path
                
            urdf_paths.append(urdf_file_path)
            urdf_types.append('mesh')
            urdf_movables.append(True) # all mesh objects are movable
           
        elif obj['type'] == 'urdf':
            try:
                category = obj['lang']
                possible_obj_path = partnet_mobility_dict[category]
            except:
                category = obj['name']
                if category == 'Computer display':
                    category = 'Display'
                possible_obj_path = partnet_mobility_dict[category]
            
            if 'reward_asset_path' not in obj.keys():
                obj_path = np.random.choice(possible_obj_path)
                if category == 'Toaster':
                    obj_path = str(103486)
                if category == 'Microwave':
                    obj_path = str(7310)
                if category == "Oven":
                    obj_path = str(101808)
                if category == 'Refrigerator':
                    obj_path = str(10638)
            else:
                obj_path = obj['reward_asset_path']
                

            urdf_file_path = osp.join(f"{data_dir}/dataset", obj_path, "mobility.urdf")
            if use_vhacd:
                new_urdf_file_path = urdf_file_path.replace("mobility.urdf", "mobility_vhacd.urdf")
                if not osp.exists(new_urdf_file_path):
                    new_urdf_file_path = preprocess_urdf(urdf_file_path)
                urdf_paths.append(new_urdf_file_path)
            else:
                urdf_paths.append(urdf_file_path)

            urdf_types.append('urdf')
            urdf_movables.append(obj.get('movable', False)) # by default, urdf objects are not movable, unless specified

        urdf_sizes.append(obj['size'])
        urdf_locations.append(parse_center(obj['center']))
        ori = obj.get('orientation', [0, 0, 0, 1])
        if type(ori) == str:
            ori = parse_center(ori)
        urdf_orientations.append(ori)
        urdf_names.append(obj['name'])
        urdf_on_tables.append(obj.get('on_table', False))
        urdf_crop_sizes.append(obj.get('is_crop_size', True))
    return urdf_paths, urdf_sizes, urdf_locations, urdf_orientations, urdf_names, urdf_types, urdf_on_tables, use_table, urdf_crop_sizes, \
        articulated_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, robot_initial_joint_angles, initial_finger_angle
            
        
#USED
def take_round_images(env, center, distance, elevation=30, azimuth_interval=30, camera_width=640, camera_height=480,
                        return_camera_matrices=False, z_near=0.01, z_far=10, save_path=None):
    camera_target = center
    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    env_prev_view_matrix, env_prev_projection_matrix = env.view_matrix, env.projection_matrix

    rgbs = []
    depths = []
    view_camera_matrices = []
    project_camera_matrices = []
    for azimuth in range(0, 360, azimuth_interval):
        delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
        delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
        camera_position = [camera_target[0] + delta_x, camera_target[1] + delta_y, camera_target[2] + delta_z]
        env.setup_camera(camera_position, camera_target, 
                            camera_width=camera_width, camera_height=camera_height)

        rgb, depth = env.render(return_depth=True)
        rgbs.append(rgb)
        depths.append(depth)
        view_camera_matrices.append(env.view_matrix)
        project_camera_matrices.append(env.projection_matrix)
    
    env.view_matrix, env.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    if not return_camera_matrices:
        return rgbs, depths
    else:
        return rgbs, depths, view_camera_matrices, project_camera_matrices
    
def take_round_images_around_object(env, object_name, distance=None, save_path=None, azimuth_interval=30, 
                                    elevation=30, return_camera_matrices=False, camera_width=640, camera_height=480, 
                                    only_object=False):
    if only_object:
        ### make all other objects invisiable
        prev_rgbas = []
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=env.id)[0][14:18]
                    prev_rgbas.append(prev_rgba)
                    p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=env.id)

                                    
    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    res = take_round_images(env, camera_target, distance, elevation=elevation, 
                             azimuth_interval=azimuth_interval, camera_width=camera_width, camera_height=camera_height, 
                             save_path=save_path, return_camera_matrices=return_camera_matrices)

    if only_object:
        cnt = 0
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=env.id)
                    cnt += 1
                    
    return res

def center_camera_at_object(env, object_name, distance=None, elevation=30, azimuth=0, camera_width=640, camera_height=480):
    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
    delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
    camera_position = [camera_target[0] + delta_x, camera_target[1] + delta_y, camera_target[2] + delta_z]
    env.setup_camera(camera_position, camera_target, 
                        camera_width=camera_width, camera_height=camera_height)

def get_pc(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.9999]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points

def get_pixel_location(proj_matrix, view_matrix, point_3d, width, height):
    # Ensure matrices are in the correct shape
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    
    # Combine the projection and view matrices
    tran_world_pix = np.matmul(proj_matrix, view_matrix)

    # Add homogeneous coordinate to the 3D point
    point_3d_h = np.append(point_3d, 1.0)
    
    # Transform the 3D point to pixel coordinates
    pixel_h = np.matmul(tran_world_pix, point_3d_h)
    
    # Normalize by the homogeneous coordinate
    pixel_h /= pixel_h[3]
    
    # Convert from normalized device coordinates to pixel coordinates
    x_ndc, y_ndc, z_ndc = pixel_h[:3]
    
    # Transform normalized device coordinates to image coordinates
    x_img = (x_ndc * 0.5 + 0.5) * width
    y_img = (1.0 - (y_ndc * 0.5 + 0.5)) * height  # Note: y-axis is inverted
    
    return int(x_img), int(y_img), z_ndc

def get_pc_in_camera_frame(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    # turn pixels to camera cooridnates
    points = np.matmul(np.linalg.inv(proj_matrix), pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    return points

    
def setup_camera_ben(client_id, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], camera_width=1920//4, camera_height=1080//4, 
                 z_near=0.01, z_far=100):
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=client_id)
    focal_length = 450 # CAMERA_INTRINSICS[0, 0]
    fov = (np.arctan((camera_height / 2) / focal_length) * 2 / np.pi) * 180
    projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, z_near, z_far, physicsClientId=client_id)
    return view_matrix, projection_matrix

def get_pc_ben(depth, view_matrix, projection_matrix, znear, zfar):
    height, width = depth.shape
    CAMERA_INTRINSICS = np.array(
        [
            [450, 0, width / 2],
            [0, 450, height / 2],
            [0, 0, 1],
        ]
    )

    T_CAMGL_2_CAM = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    depth = zfar + znear - (2.0 * depth - 1.0) * (zfar - znear)
    depth = (2.0 * znear * zfar) / depth

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - CAMERA_INTRINSICS[0, 2]) * (depth / CAMERA_INTRINSICS[0, 0])
    py = (py - CAMERA_INTRINSICS[1, 2]) * (depth / CAMERA_INTRINSICS[1, 1])
    P_cam = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    T_camgl2world = np.asarray(view_matrix).reshape(4, 4).T
    T_world2camgl = np.linalg.inv(T_camgl2world)
    T_world2cam = T_world2camgl @ T_CAMGL_2_CAM

    Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
    Ph_world = (T_world2cam @ Ph_cam.T).T
    P_world = Ph_world[:, :3]

    return P_world

def save_env(env, save_path=None, simplified=False):
    object_joint_angle_dicts = {}
    object_joint_name_dicts = {}
    object_link_name_dicts = {}
    for obj_name, obj_id in env.urdf_ids.items():
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        object_joint_angle_dicts[obj_name] = []
        object_joint_name_dicts[obj_name] = []
        object_link_name_dicts[obj_name] = []
        for link_idx in range(0, num_links):
            joint_angle = p.getJointState(obj_id, link_idx, physicsClientId=env.id)[0]
            object_joint_angle_dicts[obj_name].append(joint_angle)
            joint_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[1].decode('utf-8')
            object_joint_name_dicts[obj_name].append(joint_name)
            link_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[12].decode('utf-8')
            object_link_name_dicts[obj_name].append(link_name)

    object_base_position = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_position[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[0]

    object_base_orientation = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_orientation[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[1]

    state = {
        'object_joint_angle_dicts': object_joint_angle_dicts,
        'object_joint_name_dicts': object_joint_name_dicts,
        'object_link_name_dicts': object_link_name_dicts,
        'object_base_position': object_base_position,
        'object_base_orientation': object_base_orientation,     
    }

    
    if not simplified:
        state['urdf_paths'] = copy.deepcopy(env.urdf_paths)
        state['object_sizes'] = env.simulator_sizes
        state['robot_name'] = env.robot_name
        state['grasped_handle'] = env.grasped_handle

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    return state

def load_env(env, load_path=None, state=None, simplified=False):

    if load_path is not None:
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
    ### set env to stored object position and orientation
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state['object_base_position'].keys():
            continue
        p.resetBasePositionAndOrientation(obj_id, state['object_base_position'][obj_name], state['object_base_orientation'][obj_name], physicsClientId=env.id)

    ### set env to stored object joint angles
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state['object_joint_angle_dicts'].keys():
            continue
        
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        for link_idx in range(0, num_links):
            joint_angle = state['object_joint_angle_dicts'][obj_name][link_idx]
            p.resetJointState(obj_id, link_idx, joint_angle, physicsClientId=env.id)

    ### recover suction
    if not simplified:
        if "urdf_paths" in state:
            env.urdf_paths = state["urdf_paths"]

        if "object_sizes" in state:
            env.simulator_sizes = state["object_sizes"]

        if "robot_name" in state:
            env.robot_name = state["robot_name"]

        if "grasped_handle" in state:
            env.grasped_handle = state["grasped_handle"]

    return state

### get handle utility functions
def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def find_nearest_point_on_line(line_pt1, line_pt2, target_pt):
    line_pt1 = np.array(line_pt1).reshape(-1, 3)
    line_pt2 = np.array(line_pt2).reshape(-1, 3)
    target_pt = np.array(target_pt).reshape(-1, 3)
    
    # Step 1: Compute the vector along the line
    line_vec = line_pt2 - line_pt1
    
    # Step 2: Compute the vector from line_pt1 to target_pt
    pt_vec = target_pt - line_pt1
    
    # Step 3: Project pt_vec onto line_vec to find the projection scalar
    # dot_product(pt_vec, line_vec) / dot_product(line_vec, line_vec) gives the scalar
    # by which to multiply line_vec to get the projection vector.
    projection_scalar = np.sum(pt_vec * line_vec, axis=1) / np.sum(line_vec * line_vec)
    
    
    # Step 4: Find the nearest point on the line by scaling line_vec and adding it to line_pt1
    nearest_pt = line_pt1 + projection_scalar.reshape(-1, 1) * line_vec.repeat(len(projection_scalar), axis=0)
    
    return nearest_pt # (-1, 3)

def rotate_point_around_axis(pt, ax, theta_rad):
    """
    Rotate a point around a given axis by theta radiance.
    
    :param pt: The point to rotate (3D coordinates).
    :param ax: The rotation axis (3D unit vector).
    :param theta: The rotation angle in radians.
    :return: The rotated point's coordinates.
    """
    # Ensure ax is a unit vector
    ax = ax / np.linalg.norm(ax)
    ax = ax.reshape(-1, 3)
    
    # Rodrigues' rotation formula
    v_rot = (pt * np.cos(theta_rad) +
             np.cross(ax, pt) * np.sin(theta_rad) +
             ax * np.sum(ax.repeat(pt.shape[0], axis=0) * pt, axis=1, keepdims=True) * (1 - np.cos(theta_rad)))
    
    return v_rot

def add_sphere(position, radius=0.05, rgba=[0, 1, 1, 1]):
    sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius) 
    sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    mass = 0.1
    body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                             basePosition=position)
    return body

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_6D_to_matrix_batch(orient):

    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)
    
    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:,0]
    a2 = orient[:,1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1,1)
    b2 = a2 - (np.sum(a2*b1, axis=-1).reshape(-1,1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1,1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B x 3, 3) rotation transpose (i.e., row vectors instead of column vectors)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]

###########################################

# [Chialiang]: The following functions are from my previous projects
def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def pose_6d_to_7d(pose) -> np.ndarray:
    assert len(pose) == 6, f'pose must contain 6 elements, but got {len(pose)}'
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose = list(pos) + list(rot)

    return np.array(pose)

def pose_6d_to_7d(pose) -> np.ndarray:
    if len(pose) == 7:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def pose_7d_to_6d(pose) -> np.ndarray:
    if len(pose) == 6:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_quat(pose[3:]).as_rotvec()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def get_matrix_from_pose(pose) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7 or len(pose) == 9, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

def rot_6d_to_3d(rot) -> np.ndarray:

    rot_xy = np.asarray(rot)

    assert rot_xy.shape == (6,), f'dimension of rot should be (6,), but got {rot_xy.shape}'

    rot_xy = rot_xy.reshape(2, 3)
    rot_mat = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T 

    return R.from_matrix(rot_mat).as_rotvec()

def get_pose_from_matrix(matrix, pose_size : int = 7) -> np.ndarray:

    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)
            
    pose = list(pos) + list(rot)

    return np.array(pose)

def get_matrix_from_pos_rot(pos, rot) -> np.ndarray:
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = R.from_rotvec(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4: # x, y, z, w
        rot_m = R.from_quat(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_pos_rot_from_matrix(pose : np.ndarray) -> np.ndarray:
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot

def get_projmat_and_intrinsic(width, height, fx, fy, far, near):

  cx = width / 2
  cy = height / 2
  fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

  project_matrix = p.computeProjectionMatrixFOV(
                      fov=fov,
                      aspect=width/height,
                      nearVal=near,
                      farVal=far
                    )
  
  intrinsic = np.array([
                [ fx, 0.0,  cx],
                [0.0,  fy,  cy],
                [0.0, 0.0, 1.0],
              ])
  
  return project_matrix, intrinsic

def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):

    view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cameraEyePosition,
                    cameraTargetPosition=cameraTargetPosition,
                    cameraUpVector=cameraUpVector
                  )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm
   
    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f'cameraUpVector is parallel to z axis'
    y /= norm
    
    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic

def draw_coordinate(pose, size, color : np.ndarray=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or (len(pose) == 7) or (len(pose) == 6)

    if len(pose) == 7 or len(pose) == 6:
        pose = get_matrix_from_pose(pose)

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0], 2, 0)
    p.addUserDebugLine(origin, y, color[1], 2, 0)
    p.addUserDebugLine(origin, z, color[2], 2, 0)

def sample_point_inside_triangle(v1,v2,v3):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    while r1 + r2 >= 1:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
    r3 = 1 - r1 - r2

    # Calculate the point using barycentric coordinates
    x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
    y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
    z = r1 * v1[2] + r2 * v2[2] + r3 * v3[2]
    return [x, y, z]

def get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=0.02):
    handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
    distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
    min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
    min_distance_handle_idx = np.argmin(min_distance)
    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]
    pc_to_handle_distance = scipy.spatial.distance.cdist(link_pc, handle_pc).min(axis=1)
    handle_pc = link_pc[pc_to_handle_distance < threshold]
    # use the pointcloud of link instead of the handle itself. (partially occluded)
    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx

def get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, pc_points):
    
    cur_pos, cur_orient = cur_eef_pos, cur_eef_orient

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.02, -0.06, -0.01] 
    crop_max = [0.02, 0.06, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )
    
    within_bbox_handle_pc = pc_points[indices]
    if len(within_bbox_handle_pc) == 0:
        return 0
    score = np.sum(indices) 
    return score

def get_handle_orient(handle_pc):
    # get axis aligned bounding box of the handle pc
    min_xyz = np.min(handle_pc, axis=0)
    max_xyz = np.max(handle_pc, axis=0)
    x_range = max_xyz[0] - min_xyz[0]
    y_range = max_xyz[1] - min_xyz[1]
    z_range = max_xyz[2] - min_xyz[2]
    horizontal_range = np.max([x_range, y_range])
    vertical_range = z_range
    if horizontal_range > vertical_range:
        handle_orient = "horizontal"
    else:
        handle_orient = "vertical"
    
    return handle_orient

def pc_to_line_distance(pc, line_start, line_end):
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_vec = line_end - line_start
    pc_vecs = pc - line_start
    cross_prod = np.cross(line_vec, pc_vecs)
    distances = np.linalg.norm(cross_prod, axis=1) / np.linalg.norm(line_vec)
    return distances

def filter_pointcloud_by_line(pointcloud, point1, point2):
    """
    Filter point cloud based on line projection in xoy plane, keeping points on the same side as origin [0,0]
    
    Args:
        pointcloud: numpy array, shape [n, 3], each row is [x, y, z]
        point1: list/array, first point [x1, y1, z1] (only x,y coordinates are used)
        point2: list/array, second point [x2, y2, z2] (only x,y coordinates are used)
    
    Returns:
        numpy array: filtered point cloud
    """
    
    # Extract xy coordinates
    point1_xy = np.array(point1[:2])  # [x1, y1]
    point2_xy = np.array(point2[:2])  # [x2, y2]
    origin = np.array([0.0, 0.0])
    
    # Calculate direction vector of the line
    direction = point2_xy - point1_xy
    
    # Calculate normal vector (perpendicular to the line)
    # If direction vector is [a, b], then normal vector is [-b, a] or [b, -a]
    normal = np.array([-direction[1], direction[0]])
    
    # Ensure normal vector is not zero vector
    if np.linalg.norm(normal) == 0:
        raise ValueError("Two points are coincident or on the same vertical line, cannot form a valid dividing line")
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distance from origin to the line
    # Line equation: normal[0] * (x - x1) + normal[1] * (y - y1) = 0
    origin_distance = np.dot(normal, origin - point1_xy)
    
    # Calculate signed distance from each point in point cloud to the line
    pointcloud_xy = pointcloud[:, :2]  # Extract only x,y coordinates
    point_distances = np.dot(pointcloud_xy - point1_xy, normal)
    
    # Filter points on the same side as origin (same sign)
    if origin_distance >= 0:
        # Origin is on positive side of line, select points on positive side
        mask = point_distances >= 0
    else:
        # Origin is on negative side of line, select points on negative side
        mask = point_distances <= 0
    
    return pointcloud[mask]


def estimate_line_direction(pc):
    centered = pc - np.mean(pc, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    principal_axis /= np.linalg.norm(principal_axis)
    return principal_axis


def in_bbox(pos, bbox_min, bbox_max):
    if (pos[0] <= bbox_max[0] and pos[0] >= bbox_min[0] and \
        pos[1] <= bbox_max[1] and pos[1] >= bbox_min[1] and \
        pos[2] <= bbox_max[2] and pos[2] >= bbox_min[2]):
        return True
    return False

def draw_bbox(start, end):

    assert len(start) == 3 and len(end) == 3, f'infeasible size of position, len(position) must be 3'

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])

def radial_shift(x_coord: float, y_coord: float, noise_bounds: List[float]):
    theta = np.arctan2(y_coord, x_coord)
    theta_noise = np.random.uniform(-0.1, 0.1)
    dist = np.linalg.norm([x_coord, y_coord])
    dist_noise = np.random.uniform(noise_bounds[0],noise_bounds[1])
    theta += theta_noise
    dist += dist_noise
    perturbed_x = dist * np.cos(theta)
    perturbed_y = dist * np.sin(theta)
    return perturbed_x, perturbed_y

def get_joint_id_from_name(simulator, object_name, joint_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = i
            break

    return joint_index

def get_handle_pos(simulator, obj_name, return_median=True, handle_pts_obj_frame=None, mobility_info=None, return_info=False, target_object='handle'):
    obj_name = obj_name.lower()
    scaling = simulator.simulator_sizes[obj_name]

    # get the parent frame of the revolute joint.
    obj_id = simulator.urdf_ids[obj_name] 

    # axis in parent frame, transform everything to world frame
    if mobility_info is None:
        urdf_path = simulator.urdf_paths[obj_name]
        parent_dir = os.path.dirname(urdf_path)
        start_idx = parent_dir.find("data/dataset")
        parent_dir = parent_dir[start_idx:]
        parent_dir = os.path.join(os.environ["PROJECT_DIR"], parent_dir)
        mobility_info = json.load(open(f"{parent_dir}/mobility_v2.json", "r"))
    
    # return a list of handle points in world frame
    ret_handle_pt_list = []
    ret_joint_idx_list = []

    joint_name = None
    parent_joint_name = None
    handle_idx = 0
    all_handle_pts_object_frame = []
    for idx, joint_info in enumerate(mobility_info):
        all_parts = [part["name"] for part in joint_info["parts"]]
        if target_object in all_parts:
            all_ids = [part["id"] for part in joint_info["parts"]]
            index = all_parts.index(target_object)
            handle_id = all_ids[index]
            joint_name = "joint_{}".format(joint_info["id"])
            parent_joint_name = "joint_{}".format(joint_info["parent"])
            joint_data = joint_info['jointData']
            axis_body = np.array(joint_data["axis"]["origin"]) * scaling
            axis_dir_body = np.array(joint_data["axis"]["direction"])
            joint_limit = joint_data["limit"]
            if joint_limit['a'] > joint_limit['b']:
                axis_dir_body = -axis_dir_body

            joint_idx = get_joint_id_from_name(simulator, obj_name, joint_name) # this is the joint id in pybullet
            parent_joint_idx = get_joint_id_from_name(simulator, obj_name, parent_joint_name) # this is the joint id in pybullet
            
            parent_link_state = p.getLinkState(obj_id, parent_joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            # parent_link_state = p.getLinkState(obj_id, joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            link_urdf_world_pos, link_urdf_world_orn = parent_link_state[0], parent_link_state[1]
            # this is the transformation from the parent frame to the world frame. 
            T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
            T_body_to_world[:3, 3] = link_urdf_world_pos
            
            axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
            axis_pt2_body = np.array(axis_body) + axis_dir_body
            axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
            axis_dir_world = axis_end_world - axis_world

            # get the handle points in world frame
            if handle_pts_obj_frame is None:
                handle_obj_path = f"{parent_dir}/parts_render/{handle_id}{target_object}.obj" # NOTE: this path should be dependent on the object. 
                handle_pts, handle_faces = load_obj(handle_obj_path) # this is in object frame

                handle_pts = handle_pts * scaling
                # add more dense points around handle
                added_points = []
                for f in handle_faces:
                    v1,v2,v3 = f
                    v1 = handle_pts[v1-1]
                    v2 = handle_pts[v2-1]
                    v3 = handle_pts[v3-1]
                    a = np.linalg.norm(v1-v2)
                    b = np.linalg.norm(v2-v3)
                    c = np.linalg.norm(v3-v1)
                    s = (a+b+c) / 2
                    temp = max(0, s*(s-a)*(s-b)*(s-c))
                    surface = np.sqrt(temp)
                    num_points = surface * 1e6
                    num_points = int(num_points)
                    num_points = np.clip(num_points, 0, 5)
                    added_points.extend([sample_point_inside_triangle(v1,v2,v3) for _ in range(num_points)])

                if added_points != []:
                    added_points = np.array(added_points)
                    handle_pts = np.concatenate((handle_pts, added_points), axis=0)
                    
                all_handle_pts_object_frame.append(handle_pts)
                    
            else:
                handle_pts = handle_pts_obj_frame[handle_idx]
            
            
            # transform this to the world frame using the object *base*'s position and orientation
            handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1) # 3 x N
            if return_median:
                handle_point_median = np.median(handle_points_world, axis=1)
            else:
                handle_point_median = handle_points_world.T

            # find the projection of the handle point to the rotation axis, in world frame. 
            project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
            # p.addUserDebugLine(project_on_rotation_axis, handle_point_median, [1, 0, 0], 25, 0)

            # TODO: GPT can parse the mobility.json to get the joint name. 
            joint_info = p.getJointInfo(obj_id, joint_idx, physicsClientId=simulator.id)
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                rotation_angle = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0] # NOTE: this joint id should be dependent on the object urdf.
                rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
                rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
            elif joint_type == p.JOINT_PRISMATIC:
                translation = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0]
                rotated_handle_pt = handle_point_median + axis_dir_world * translation
                
            # import pdb; pdb.set_trace()
            # rotated_handle_pt = handle_points_world.T

            if return_median:
                ret_handle_pt_list.append(rotated_handle_pt.flatten())
            else:
                ret_handle_pt_list.append(rotated_handle_pt)
            ret_joint_idx_list.append(joint_idx)
            
            handle_idx += 1
            
    if return_info:
        return ret_handle_pt_list, ret_joint_idx_list, all_handle_pts_object_frame, mobility_info
    
    return ret_handle_pt_list, ret_joint_idx_list


def get_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name 
    link_com, all_pc = render_to_get_link_com(simulator, object_name, urdf_link_name)

    return all_pc

def render_to_get_link_com(simulator, object_name, urdf_link_name):    
    ### make all other objects invisiable
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = simulator.view_matrix, simulator.projection_matrix
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30

    ### get a round of images of the target object
    rgbs, depths, view_matrices, projection_matrices = take_round_images(
        simulator, camera_target, distance, elevation, 
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10,
        return_camera_matrices=True)

    ### make the target link invisiable
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    # import pdb; pdb.set_trace()
    prev_link_rgba = p.getVisualShapeData(obj_id, link_id, physicsClientId=simulator.id)[0][14:18]
    p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### get a round of images of the target object with link invisiable
    rgbs_link_invisiable, depths_link_invisible, _, _ = take_round_images(
        simulator, camera_target, distance, elevation,
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10, return_camera_matrices=True
    )

    ### use subtraction to get the link mask
    max_num_diff_pixels = 0
    best_idx = 0
    for idx, (depth, depth_) in enumerate(zip(depths, depths_link_invisible)):
        diff_image = np.abs(depth - depth_)
        diff_pixels = np.sum(diff_image > 0)
        if diff_pixels > max_num_diff_pixels:
            max_num_diff_pixels = diff_pixels
            best_idx = idx
    best_mask = np.abs(depths[best_idx] - depths_link_invisible[best_idx]) > 0
    # best_mask = np.any(best_mask)


    ### get the link mask center
    center = ndimage.measurements.center_of_mass(best_mask)
    center = [int(center[0]), int(center[1])]

    ### back project the link mask center to get the link com in 3d coordinate
    best_pc = get_pc(projection_matrices[best_idx], view_matrices[best_idx], depths[best_idx], camera_width, camera_height)
    
    pt_idx = center[0] * camera_width + center[1]
    link_com = best_pc[pt_idx]
    best_pc = best_pc.reshape((camera_height, camera_width, 3))
    all_pc = best_pc[best_mask]


    ### reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    p.changeVisualShape(obj_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=simulator.id)

    cnt = 0
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    ### add a safety check here in case the rendering fails
    bounding_box = get_bounding_box_link(simulator, object_name, urdf_link_name)
    if not in_bbox(link_com, bounding_box[0], bounding_box[1]):
        link_com = (bounding_box[0] + bounding_box[1]) / 2

    return link_com, all_pc

def get_bounding_box_link(simulator, object_name, link_name):
    object_name = object_name.lower()
    link_id = get_link_id_from_name(simulator, object_name, link_name)
    object_id = simulator.urdf_ids[object_name]
    return simulator.get_aabb_link(object_id, link_id)

def get_link_id_from_name(simulator, object_name, link_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[12].decode("utf-8") == link_name:
            joint_index = i
            break

    return joint_index

if __name__ == '__main__':
    
    path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e5/yufei/projects/ibm/objaverse_utils/data/obj/6d9c1aa964be4f7881d89cd6b427296c____Small_house_with_wrecked_car"
    path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e5/yufei/projects/ibm/objaverse_utils/data/obj/94ccd348a1424defaea6efcd1d3418a6____Plastic_monster_toy."
    path = "objaverse_utils/data/obj/0/006_mustard_bottle/tsdf/"
    run_vhacd(path, normalized=False, obj_name='textured')
    res = obj_to_urdf(path, 1, vhacd=True, normalized=False, obj_name='textured')