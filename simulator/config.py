import math
import pickle
import os
import warnings


class Config:
    world_type_cubes_and_spheres = 0
    world_type_maze = 1
    collision_behavior_skip = 0     # Does not do anything if path would collide
    collision_behavior_crop = 1     # Moves the maximal possible distance towards the original goal
    glob_frontier_selection_closest = 0     # Euclidean distance
    glob_frontier_selection_random = 1

    def __init__(self):
        """ Sets default values for every param"""
        # world
        self.world_size_x = 50  # m
        self.world_size_y = 50  # m
        self.voxel_size = 0.2  # m

        # world population
        self.world_type = Config.world_type_cubes_and_spheres
        # cubes_and_spheres
        self.cs_n_cubes = [10, 40]  # min, max
        self.cs_n_spheres = [10, 40]
        self.cs_cube_side_lengths = [1, 10]  # m
        self.cs_spheres_radii = [1, 5]  # m
        # maze
        self.mz_n_walls = [100, 200]
        self.mz_wall_strength = 0.5  # m
        self.mz_wall_length = [30, 45]  # m
        self.mz_wall_spacing = 5  # m

        # robot
        self.local_submap_size_x = 10 / self.voxel_size  # voxels
        self.local_submap_size_y = 10 / self.voxel_size  # voxels
        self.velocity_max = 1  # m/s
        self.acceleration = 1  # m/s2
        self.yaw_rate_max = math.pi  # rad/s
        self.yaw_acceleration = math.pi * 2  # rad/s2
        self.camera_range = 5  # m
        self.camera_fov = 90  # deg, can use 360 for lidar
        self.camera_rate = 5  # Hz
        # robot startup
        self.initial_safety_margin = 1  # m, minimum distance to walls at initialization
        self.initial_yaw = None    # [0, 2pi], set None for random
        self.initial_margin_observed = True     # Whether to clear the safety margin on startup
        # robot behavior
        self.execute_unsafe_trajectories = True     # What to do if requested path is potentially unsafe
        self.collision_behavior = Config.collision_behavior_skip

        # global planner
        self.glob_radius = 2    # m, distance within frontier to sample goal poses from
        self.glob_frontier_selection = Config.glob_frontier_selection_closest
        self.glob_min_frontier_size = 3     # voxels
        self.glob_frontier_consolidation_radius = 1  # m
        self.rrt_expand_dis = 3  # m
        self.rrt_goal_sample_rate = 20  # % of samples from goal
        self.rrt_max_iter = 10000  # maximum iterations
        self.rrt_min_iter = 100  # minimum iterations (refines path)
        self.rrt_connect_dist = 10  # m

        # saving images
        self.save_images = False    # Whether to save image data
        self.images_folder = '../images/'

    def save_to_file(self, filepath):
        """ Utility function to save/load configs, expected as /path/to/dir/filename """
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        pickle.dump([self], open(filepath + ".p", "wb"))

    def load_from_file(self, filepath):
        """ Utility function to save/load configs """
        if not os.path.isfile(filepath + ".p"):
            warnings.warn("Cannot load '%s': is not an existing file!" % filepath + ".p")
            return
        data = pickle.load(open(filepath + ".p", "rb"))
        # For backwards compatibility
        current = self.__dict__
        for key in data[0].__dict__:
            if key in current:
                current[key] = data[0].__dict__[key]
        self.__dict__.update(current)
