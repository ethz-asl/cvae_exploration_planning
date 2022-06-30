import math
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from cvae_exploration_planning.simulator.config import Config
from cvae_exploration_planning.simulator.simulator import Simulator


""" Create random worlds and starting poses and save them """
def get_config(is_maze):
    cfg = Config()  # sets up defaults

    # Custom params.
    cfg.voxel_size = 0.2 
    cfg.initial_safety_margin = 2

    if is_maze:
        # World Maze
        cfg.world_type = Config.world_type_maze
        cfg.mz_n_walls = [100, 200]
        cfg.mz_wall_strength = 0.5  # m
        cfg.mz_wall_length = [30, 45]  # m
        cfg.mz_wall_spacing = 5  # m
    else:
        # World Cube and Sphere
        cfg.world_type = Config.world_type_cubes_and_spheres
        cfg.cs_n_cubes = [10, 30]  # min, max
        cfg.cs_n_spheres = [10, 30]
        cfg.cs_cube_side_lengths = [1, 8]  # m
        cfg.cs_spheres_radii = [1, 4]  # m
    
    return cfg

def explore_worlds():
    # Config
    is_maze = True
    cfg = get_config(is_maze)

    # A little nicer plotting
    free = np.array([1, 1, 1, 1])
    occ = np.array([0, 0, 0, 1])
    world_cmp = ListedColormap(np.vstack((free, occ)))

    # setup the simulator
    sim = Simulator(cfg)

    # Propose different worlds.
    plt.figure()
    sim.reset_world()
    sim.reset_robot()
    while True:
        plt.clf()
        world_map = sim.get_world_map()
        x, y, yaw = sim.get_robot_pose()
        plt.imshow(world_map, cmap=world_cmp, interpolation='none')
        plt.arrow(y, x, math.sin(yaw) / cfg.voxel_size,
                  math.cos(yaw) / cfg.voxel_size, width=0.3 / cfg.voxel_size,
                  color='r') 
        plt.draw()
        plt.pause(0.001)
        observable = sim.get_explorable_area()
        print("Displaying the generated world map and initial robot pose.\n"
              "From this starting point, %.2f%s of the world (%i voxels) are observable."
              % (observable / np.shape(world_map)[0] / np.shape(world_map)[
            1] * 100,'%', observable))

        print("Enter '1' to create a new world, '2' to reset the start pose, '3' to exit the demo, or any other string to save the world.")
        k = input()
        if k == "1":
            sim.reset_world()
            sim.reset_robot()
        elif k == "2":
            sim.reset_robot()
        elif k == "3":
            break
        else:
            file_name = os.path.join("worlds", k)
            np.savetxt(file_name + '.txt', np.array((x, y, yaw)))
            plt.savefig(file_name + '.png')
            sim.save_state_to_file(file_name)
            print(f"Saved world to '{os.path.abspath(file_name)}'.")

if __name__ == "__main__":
    explore_worlds()
