try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
    ou.setLogLevel(ou.LOG_ERROR)
except ImportError:
    print("ompl is not installed, data generation code won't work")
    pass
import pybullet as p
import pybullet_ompl.utils as utils
import time
from itertools import product
import copy
import numpy as np
import random


INTERPOLATE_NUM = 100
DEFAULT_PLANNING_TIME = 5

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id, control_joint_idx=None, object_id=None, env=None) -> None:
        # Public attributes
        self.id = id
        self.env = env
        self.physics_id = env.id
        
        # prune fixed joints
        all_joint_num = p.getNumJoints(id, physicsClientId=env.id)
        all_joint_idx = list(range(all_joint_num))
        if control_joint_idx is None:
            joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        else:
            joint_idx = control_joint_idx
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        # print("pbompl joint number: ", self.num_dim)
        # print("pbompl joint idx: ", self.joint_idx)
        self.joint_bounds = []
        self.get_joint_bounds()

        self.object_id = object_id

        if object_id is not None:
            object_init_pos, object_init_orn = p.getBasePositionAndOrientation(object_id, physicsClientId=self.env.id)
            robot_init_pos, robot_init_orn = env.robot.get_pos_orient(env.robot.right_end_effector)
            world_to_robot = p.invertTransform(robot_init_pos, robot_init_orn)
            object_in_robot = p.multiplyTransforms(world_to_robot[0], world_to_robot[1], object_init_pos, object_init_orn)
            self.object_in_robot_pos, self.object_in_robot_orn = object_in_robot[0], object_in_robot[1]

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx, physicsClientId=self.env.id)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id, physicsClientId=self.env.id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds

            # TODO: the indicies may be different for a mobile manipulator
            if not self.env.mobile:
                if i in self.joint_idx:
                    if i != 9 and i != 10:
                        delta = 0.05 * (high - low)
                        low += delta
                        high -= delta

            if low < high:
                self.joint_bounds.append([low, high])
                
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state
        if self.object_id is not None:
            robot_target_pos, robot_target_orn = self.env.robot.get_pos_orient(self.env.robot.right_end_effector)
            object_target_pos, objectr_target_orientation = p.multiplyTransforms(robot_target_pos, robot_target_orn, self.object_in_robot_pos, self.object_in_robot_orn)
            p.resetBasePositionAndOrientation(self.object_id, object_target_pos, objectr_target_orientation, physicsClientId=self.env.id)

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state
        if self.object_id is not None:
            robot_target_pos, robot_target_orn = self.env.robot.get_pos_orient(self.env.robot.right_end_effector)
            object_target_pos, objectr_target_orientation = p.multiplyTransforms(robot_target_pos, robot_target_orn, self.object_in_robot_pos, self.object_in_robot_orn)
            p.resetBasePositionAndOrientation(self.object_id, object_target_pos, objectr_target_orientation, physicsClientId=self.env.id)

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0, physicsClientId=self.env.id)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, obstacles = [], allow_collision_links=[], allow_collision_robot_link_pairs=[], object_id=None,
                 interpolation_num=None, goal_allow_collide=False) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
            object_id: id of the object holded by the robot. Optional.
        '''
        
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.allow_collision_links = allow_collision_links
        self.allow_collision_robot_link_pairs = allow_collision_robot_link_pairs
        self.interpolation_num = interpolation_num
        self.goal_allow_collide = goal_allow_collide

        self.space = PbStateSpace(robot.num_dim)
        self.start = copy.deepcopy(self.robot.get_cur_state())
        self.goal = copy.deepcopy(self.robot.get_cur_state())

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.joint_bounds
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])
        self.set_obstacles(obstacles)
        # notice this must be called after set_obstacles
        self.set_object(object_id)
        self.set_planner("RRTConnect") # RRTConnect by default

        random.seed(time.time_ns() % 2**32)
        np.random.seed(time.time_ns() % 2**32)

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)
    
    def set_object(self, object_id):
        self.object_id = object_id

        if object_id is not None:
            self.check_body_pairs = self.check_body_pairs + list(product([object_id], self.obstacles))

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set
        
        # ignore possible location in the initial state
        cur_state = self.state_to_list(state)
        if np.allclose(self.start, cur_state):
            return True
        
        if self.goal_allow_collide:
            if np.allclose(cur_state, self.goal):
                return True
            
        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2, p_id=self.robot.env.id):
                # print("collision between robot link {} and {}".format(link1, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2, p_id=self.robot.env.id):
                # print("collision between body {} and {}".format(body1, body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True):
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        for allow_collision_pair in self.allow_collision_robot_link_pairs:
            self.check_link_pairs.remove(allow_collision_pair)
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.joint_idx) if not item in self.allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "ABITstar":
            self.planner = og.ABITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME, smooth_path=True):
        '''
        plan a path to gaol from the given robot start state
        '''
        # print("start_planning")
        # print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        if solved:
            # print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            if smooth_path:
                sol_path_geometric = self.smooth_path(sol_path_geometric)
            if self.interpolation_num is None:
                self.interpolation_num = INTERPOLATE_NUM
            sol_path_geometric.interpolate(self.interpolation_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # if smooth_path:
            #     sol_path_list = self.smooth_python_path(sol_path_list)
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME, smooth_path=True):
        '''
        plan a path to gaol from current robot state
        '''
        self.goal = list(goal)
        start = self.robot.get_cur_state()
        res, path = self.plan_start_goal(start, goal, allowed_time=allowed_time, smooth_path=smooth_path)
        return res, path
    

    def smooth_random_indices(self, N):
        # First sample two indices without replacement
        idx0 = random.randint(0, N - 1)
        # This is a little trick to just not pick the same index twice
        idx1 = random.randint(0, N - 2)
        if idx1 >= idx0:
            idx1 += 1
        # Reset the variable names to be in order
        idx0, idx1 = (idx0, idx1) if idx1 > idx0 else (idx1, idx0)
        return idx0, idx1
    
    def smooth_steer_to(self, start, end):
        """
        I don't like the name steer_to but other people use it so whatever
        """
        # Check which joint has the largest movement
        which_joint = np.argmax(np.abs(end - start))
        num_steps = int(np.ceil(np.abs(end[which_joint] - start[which_joint]) / 0.1))
        return np.linspace(start, end, num=num_steps)

    def smooth_path(self, path):
        '''
        Smooth path using OMPL's path simplifier
        '''
        # create a path simplifier
        ps = og.PathSimplifier(self.ss.getSpaceInformation())
        # simplify the path
        try:
            success1 = ps.partialShortcutPath(path)
            success2 = ps.ropeShortcutPath(path)
        except:
            success1 = ps.shortcutPath(path)
        success3 = ps.smoothBSpline(path)
        # if not success1 or not success2 or not success3:
        #     print("Failed to simplify path")
        return path

    def smooth_python_path(self, path):
        path = np.asarray(path)
        indexed_path = list(zip(path, range(len(path))))
        checked_pairs = set()

        for _ in range(len(path)):
            idx0, idx1 = self.smooth_random_indices(len(indexed_path))
            start, idx_start = indexed_path[idx0]
            end, idx_end = indexed_path[idx1]
            # Skip if this pair was already checked
            if (idx_start, idx_end) in checked_pairs:
                continue

            # The collision check resolution should never be smaller
            # than the original path was, so use the indices from the original
            # path to determine how many collision checks to do
            shortcut_path = self.smooth_steer_to(start, end)
            good_path = True
            # Check the shortcut
            for q in shortcut_path:
                if not self.is_state_valid(q):
                    good_path = False
                    break
            if good_path:
                indexed_path = indexed_path[: idx0 + 1] + indexed_path[idx1:]

            # Add the checked pair into the record to avoid duplicates
            checked_pairs.add((idx_start, idx_end))

        # TODO move this into a test suite instead of a runtime check
        assert np.allclose(path[0], indexed_path[0][0])
        assert np.allclose(path[-1], indexed_path[-1][0])
        path = [p[0] for p in indexed_path]

        return path
    
    # def smooth_and_interpolate_path(self, path, timesteps=INTERPOLATE_NUM):
    #     curve = smooth_cubic(
    #         path,
    #         lambda q: not self._not_in_collision(q),
    #         np.radians(3) * np.ones(7),
    #         self.robot_type.VELOCITY_LIMIT,
    #         self.robot_type.ACCELERATION_LIMIT,
    #     )
    #     ts = (curve.x[-1] - curve.x[0]) / (timesteps - 1)
    #     return [curve(ts * i) for i in range(timesteps)]


    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i],force=5 * 240., physicsClientId=self.robot.env.id)
            else:
                self.robot.set_state(q)
            p.stepSimulation(physicsClientId=self.robot.env.id)
            time.sleep(0.01)



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]
    