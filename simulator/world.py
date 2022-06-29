import numpy as np
import math
import warnings
import random

from cvae_exploration_planning.simulator.config import Config


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        self.map_gt = None  # 2D ground truth map: 0 = free space, 1 = occupied, stored as map[x,y]
        self.map_size_x = None  # in voxels
        self.map_size_y = None  # in voxels

    def create_random_world(self):
        """ Overwrite the current world with a newly random generated one. """
        self.map_size_x = math.floor(self.cfg.world_size_x / self.cfg.voxel_size) - 1
        self.map_size_y = math.floor(self.cfg.world_size_y / self.cfg.voxel_size) - 1
        self.map_gt = np.zeros((self.map_size_x + 1, self.map_size_y + 1), dtype=int)
        # boundaries are occupied
        self.map_gt[:, 0] = 1
        self.map_gt[:, -1] = 1
        self.map_gt[0, :] = 1
        self.map_gt[-1, :] = 1

        # populate world according to model
        if self.cfg.world_type is Config.world_type_cubes_and_spheres:
            self.populate_world_cubes_and_spheres()
        elif self.cfg.world_type is Config.world_type_maze:
            self.populate_world_maze()
        else:
            warnings.warn("Unknown world_type '%s'!" % self.cfg.world_type)

    def populate_world_maze(self):
        """ simple world model with walls connecting to each other """
        n_walls = random.randint(self.cfg.mz_n_walls[0], self.cfg.mz_n_walls[1])
        thickness = math.floor(self.cfg.mz_wall_strength / self.cfg.voxel_size)
        spacing = math.floor(self.cfg.mz_wall_spacing / self.cfg.voxel_size)
        for i in range(0, n_walls):
            x = random.randint(0, self.map_size_x)
            y = random.randint(0, self.map_size_y)
            x = math.floor(x / spacing) * spacing
            y = math.floor(y / spacing) * spacing
            direction = random.randint(0, 3)
            j_max = random.randint(self.cfg.mz_wall_length[0], self.cfg.mz_wall_length[1]) / self.cfg.voxel_size
            j = 0
            if direction == 0:
                while self.map_gt[x + j, y] == 0 and j < j_max:
                    self.map_gt[x + j, y:min(y + thickness, self.map_size_y)] = 1
                    j = j + 1
            elif direction == 1:
                while self.map_gt[x - j, y] == 0 and j < j_max:
                    self.map_gt[x - j, y:min(y + thickness, self.map_size_y)] = 1
                    j = j + 1
            elif direction == 2:
                while self.map_gt[x, y + j] == 0 and j < j_max:
                    self.map_gt[x:min(x + thickness, self.map_size_x), y + j] = 1
                    j = j + 1
            else:
                while self.map_gt[x, y - j] == 0 and j < j_max:
                    self.map_gt[x:min(x + thickness, self.map_size_x), y - j] = 1
                    j = j + 1

    def populate_world_cubes_and_spheres(self):
        """ Simple world model with some cubes and some spheres """
        n_spheres = random.randint(self.cfg.cs_n_spheres[0], self.cfg.cs_n_spheres[1])
        n_cubes = random.randint(self.cfg.cs_n_cubes[0], self.cfg.cs_n_cubes[1])
        for i in range(0, n_spheres):
            x = random.randint(0, self.map_size_x)
            y = random.randint(0, self.map_size_y)
            r = random.uniform(self.cfg.cs_spheres_radii[0], self.cfg.cs_spheres_radii[1]) / self.cfg.voxel_size
            xx, yy = np.mgrid[:self.map_size_x + 1, :self.map_size_y + 1]
            sphere = (xx - x) ** 2 + (yy - y) ** 2 < r ** 2
            self.map_gt = np.maximum(self.map_gt, sphere)
        for i in range(0, n_cubes):
            x = random.randint(0, self.map_size_x)
            y = random.randint(0, self.map_size_y)
            x_len = random.randint(math.floor(self.cfg.cs_cube_side_lengths[0] / self.cfg.voxel_size),
                                   math.floor(self.cfg.cs_cube_side_lengths[1] / self.cfg.voxel_size))
            y_len = random.randint(math.floor(self.cfg.cs_cube_side_lengths[0] / self.cfg.voxel_size),
                                   math.floor(self.cfg.cs_cube_side_lengths[1] / self.cfg.voxel_size))
            self.map_gt[x:min(x + x_len, self.map_size_x), y:min(y + y_len, self.map_size_y)] = 1
