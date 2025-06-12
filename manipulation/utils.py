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
from objaverse_utils.utils import text_to_uid_dict, partnet_mobility_dict, sapaien_cannot_vhacd_part_dict
from typing import List, Optional

# Chialiang
from scipy.spatial.transform import Rotation as R

workspace = os.environ['PROJECT_DIR']
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

def build_up_env(task_config=None, solution_path=None, task_name=None, restore_state_file=None, return_env_class=False, 
                    render=False, randomize=False, 
                    obj_id=0,  **kwargs,
                ):
    
    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = task_name
    save_config['restore_state_file'] = restore_state_file
    save_config['gui'] = render
    save_config['randomize'] = randomize
    save_config['obj_id'] = obj_id
    save_config['task_name'] = task_name
    for key, value in kwargs.items():
        save_config[key] = value

    ### you might want to restore to a specific state
    module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    env_class = getattr(module, task_name)
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


def parse_config(config, use_bard=True, obj_id=None, use_gpt_size=True, use_vhacd=True):
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
                

            if 'urdf' not in obj_path:
                urdf_file_path = osp.join(f"{data_dir}/dataset", obj_path, "mobility.urdf")
                if use_vhacd:
                    new_urdf_file_path = urdf_file_path.replace("mobility.urdf", "mobility_vhacd.urdf")
                    if not osp.exists(new_urdf_file_path):
                        new_urdf_file_path = preprocess_urdf(urdf_file_path)
                    urdf_paths.append(new_urdf_file_path)
                else:
                    urdf_paths.append(urdf_file_path)
            else:
                urdf_paths.append(obj_path)

            # print("obj_path: ", obj_path)
            # print("obj_path: ", obj_path)
            # print("obj_path: ", obj_path)
            # import pdb; pdb.set_trace()

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
        articulated_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, robot_initial_joint_angles
            
        

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


def save_env(env, save_path=None):
    object_joint_angle_dicts = {}
    object_joint_name_dicts = {}
    object_link_name_dicts = {}
    robot_name = env.robot_name
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
        "urdf_paths": copy.deepcopy(env.urdf_paths),
        "object_sizes": env.simulator_sizes,
        'robot_name': env.robot_name,
        "grasped_handle": env.grasped_handle,
    }

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    return state

def load_env(env, load_path=None, state=None):

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

def piecewise_uniform_sample(low: float, high: float) -> float:
    """
    Samples from a piece-wise uniform distribution of [low,high]+[-high, -low]
    """
    is_negative = np.random.uniform(0,1) <= 0.5
    if is_negative:
        return np.random.uniform(-high, -low)
    else:
        return np.random.uniform(low, high)

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

if __name__ == '__main__':
    
    # path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e5/yufei/projects/ibm/objaverse_utils/data/obj/6d9c1aa964be4f7881d89cd6b427296c____Small_house_with_wrecked_car"
    # path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e5/yufei/projects/ibm/objaverse_utils/data/obj/94ccd348a1424defaea6efcd1d3418a6____Plastic_monster_toy."
    # path = "objaverse_utils/data/obj/0/006_mustard_bottle/tsdf/"
    # run_vhacd(path, normalized=False, obj_name='textured')
    # res = obj_to_urdf(path, 1, vhacd=True, normalized=False, obj_name='textured')
    
    urdf = "/project_data/held/yufeiw2/RoboGen_sim2real/data/dataset/21473/mobility.urdf"
    preprocess_urdf(urdf)