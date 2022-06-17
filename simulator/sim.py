import numpy as np
import pickle
import os
import warnings
import math
from scipy import ndimage

from simulator.config import Config
from simulator.world import World
from simulator.robot import Robot
from planning.global_planner import GlobalPlanner


class ActionResponse:
    def __init__(self):
        self.collides = False
        self.is_safe = False
        self.path_length = 0
        self.execution_time = 0
        self.newly_observed_voxels = 0
        self.iterations = 0


class Simulator:
    def __init__(self, cfg):
        self._cfg = cfg
        self._world = World(cfg)
        self._robot = Robot(cfg)

    def reset_world(self):
        """ Reset the current world with a random model """
        self._world.create_random_world()

    def reset_robot(self):
        """ Find random initial position and take initial measurement """
        self._robot.initialize_in_world(self._world)

    def get_world_map(self):
        """ Return the ground truth environment """
        return self._world.map_gt

    def get_robot_map(self):
        """ Return the robot map, 0 = free, 1 = occupied, 2 = unobserved """
        return self._robot.observed_map

    def get_robot_local_submap(self):
        """ Get a cropped out map of the area around the robot"""
        return self._robot.get_local_submap()

    def get_robot_pose(self):
        """ Return the robot position in voxel coordinates and yaw in rad """
        return self._robot.position_x, self._robot.position_y, self._robot.yaw

    def move_to(self, x, y, yaw):
        """ Orders the robot x,y relative to its current position and face yaw, returns an ActionResponse. """
        response = ActionResponse()
        collided_at, is_safe = self._robot.check_move_feasible(x, y)
        response.collides = collided_at is not None
        response.is_safe = is_safe
        # Handle how to deal with collision here
        if (not response.is_safe) and (not self._cfg.execute_unsafe_trajectories):
            # response.collides = False
            return response
        if response.collides:
            if self._cfg.collision_behavior == Config.collision_behavior_skip:
                return response
            elif self._cfg.collision_behavior == Config.collision_behavior_crop:
                scale = max(collided_at - 0.1, 0)  # 10% safety margin
                x *= scale
                y *= scale
        # normalize yaw
        yaw = np.mod(yaw, 2 * math.pi)
        if yaw < 0:
            yaw += 2 * math.pi
        # Execute action
        unknown_voxels = np.sum(self._robot.observed_map == 2)
        distance, time = self._robot.move_to(x, y, yaw)
        response.execution_time = time
        response.path_length = distance
        response.newly_observed_voxels = unknown_voxels - np.sum(self._robot.observed_map == 2)
        return response

    def get_explorable_area(self):
        """ Returns the total number of observable voxels (orthogonally connected free space from the robot start) """
        mask = self._world.map_gt == 0
        labeled_image, _ = ndimage.measurements.label(mask)
        mask = labeled_image == labeled_image[int(self._robot.position_x), int(self._robot.position_y)]
        mask = ndimage.binary_dilation(mask)  # To include surface voxel, which are also observable
        return np.sum(mask)

    def get_explored_area(self):
        """ Returns the total number of observed voxels in the robot map """
        mask = self._robot.observed_map != 2
        return np.sum(mask)

    def call_global_planner(self, verify_path=False):
        """ Moves the robot to a new global starting pose when stuck. Just uses a closest frontier planner.
         verify_path: True compute feasible path with RRT*, False teleport the robot. """
        result = GlobalPlanner(self._cfg, self._robot).plan(verify_path)
        if not result.success:
            # No more frontiers left
            return result

        # Apply result
        self._robot.position_x = result.x
        self._robot.position_y = result.y
        self._robot.yaw = result.yaw

        self._robot.take_sensor_measurement()
        return result

    def is_in_local_minimum(self):
        """ Returns true if there are no observable frontiers in the current submap. """
        return GlobalPlanner(self._cfg, self._robot).is_local_minimum(
            self._robot.get_local_submap(), [self._cfg.local_submap_size_x / 2,
                                             self._cfg.local_submap_size_y / 2])

    def save_state_to_file(self, filepath):
        """ Save the state variables into the path, expected as /path/to/dir/filename """
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        pickle.dump([self], open(filepath + ".p", "wb"))

    def load_state_from_file(self, filepath):
        """ Load previously saved sim state """
        if not os.path.isfile(filepath + ".p"):
            warnings.warn("Cannot load '%s': is not an existing file!" % filepath + ".p")
            return
        print(filepath)
        data = pickle.load(open(filepath + ".p", "rb"))
        cfg = self._cfg
        self.__dict__.update(data[0].__dict__)
        self._cfg = cfg
        self._world.cfg = cfg
        self.robot.cfg = cfg
        self.robot.world.cfg = cfg

    @property
    def robot(self):
        return self._robot

    @property
    def config(self):
        return self._cfg
