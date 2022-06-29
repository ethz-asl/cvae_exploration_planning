from time import sleep
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from cvae_exploration_planning.simulator.config import Config
from cvae_exploration_planning.simulator.simulator import Simulator
from cvae_exploration_planning.planning.baseline_planner import LocalSamplingPlanner


def demo():
    # define the config for the entire simulation
    cfg = Config()  # sets up defaults
    cfg.voxel_size = 0.2  # set custom params
    cfg.world_type = cfg.world_type_maze
    cfg.execute_unsafe_trajectories = False

    # setup the simulator
    sim = Simulator(cfg)

    # use the sim api to command the world and robot
    sim.reset_world()
    sim.reset_robot()

    # visualize the world and robot pose
    world_map = sim.get_world_map()
    x, y, yaw = sim.get_robot_pose()

    # a little nicer plotting
    free = np.array([1, 1, 1, 1])
    occ = np.array([0, 0, 0, 1])
    unknown = np.array([0.85, 0.9, 0.95, 1])
    world_cmp = ListedColormap(np.vstack((free, occ)))
    robot_cmp = ListedColormap(np.vstack((free, occ, unknown)))
    plt.figure(figsize=(20, 20))  # large map so we can see every pixel

    # Display the generated world.
    plt.imshow(world_map, cmap=world_cmp, interpolation='none')
    plt.arrow(y, x, math.sin(yaw) / cfg.voxel_size,
              math.cos(yaw) / cfg.voxel_size, width=0.3 / cfg.voxel_size,
              color='r')  # coordinate convention: x,y is in image coordinates, i.e. x = down, y = right

    plt.draw()
    plt.pause(0.001)

    # Using some of the ground truth data from the simulator.
    observable = sim.get_explorable_area()
    print("Displaying the generated world map and initial robot pose.\n"
          "From this starting point, %.2f%s of the world (%i voxels) are observable."
          % (observable / np.shape(world_map)[0] / np.shape(world_map)[1] * 100,
             '%', observable))

    # Setup a planner for sampling based local exploration. The planner internally uses the robot to access information about the known state of the robot in simulation.
    sampling_steps = 20
    action_bounds = [cfg.local_submap_size_x /
                     2, cfg.local_submap_size_y/2, math.pi]
    count_only_feasible_samples = True
    use_normalized_gain = True
    sampling_yaw = 1
    planner = LocalSamplingPlanner(sim.robot, sampling_steps,
                                   sampling_yaw, action_bounds,
                                   count_only_feasible_samples,
                                   use_normalized_gain)

    # Some interactive random movements and visualization
    print("Enter '1' to let the robot move once, '2' to let the robot move continuosly, or '3' to exit the demo:")
    autoplay = False
    while True:
        k = None
        if not autoplay:
            k = input()
        else:
            k = "1"
            sleep(1)
        if k == "2":
            autoplay = True
        elif k == "3":
            break
        elif k == "1":
            # Get the desired action from the planner.
            plan, _, _ = planner.plan()

            # Let the robot move in simulation.
            response = sim.move_to(plan.x, plan.y, plan.yaw)
            # Plot the new robot state
            robot_map = sim.get_robot_map()
            plt.clf()
            plt.imshow(robot_map, cmap=robot_cmp, interpolation='none')
            x_new, y_new, yaw_new = sim.get_robot_pose()
            plt.arrow(y_new, x_new, math.sin(yaw_new) / cfg.voxel_size,
                      math.cos(yaw_new) / cfg.voxel_size,
                      width=0.3 / cfg.voxel_size, color='r')
            dx = np.linspace(x, x_new)
            dy = np.linspace(y, y_new)
            plt.plot(dy, dx, linestyle=':', color='orange')
            plt.arrow(y, x, 0.6 * math.sin(yaw) / cfg.voxel_size,
                      0.6 * math.cos(yaw) / cfg.voxel_size,
                      width=0.2 / cfg.voxel_size, color='orange')
            plt.draw()
            plt.pause(0.001)
            print(
                f"Found a target after evaluating {plan.total_samples} ({plan.feasible_samples} feasible) samples. Discovered {response.newly_observed_voxels} new voxels in {response.execution_time}s flight time.\n")
            x = x_new
            y = y_new
            yaw = yaw_new

            # If the robot is in a local minimum we can use a global planner to relocate it.
            if sim.is_in_local_minimum():
                sim.call_global_planner()
                print(
                    "Robot was stuck in a local minimum and moved to a new global exploration site.")
            if not autoplay:
                print(
                    "Enter '1' to let the robot move once, '2' to let the robot move continuosly, or '3' to exit the demo:")
        else:
            print("Enter '1' to let the robot move once, '2' to let the robot move continuosly, or '3' to exit the demo:")


if __name__ == '__main__':
    demo()
