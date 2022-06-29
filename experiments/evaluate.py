#!/usr/bin/env python
from ast import parse
from random import sample
import time
import datetime
from planning.baseline_nbvp import LocalNBVPlanner
from simulator.config import Config
from simulator.sim import Simulator
from planning.baseline_planner import LocalSamplingPlanner
from planning.policy_planner import PolicyPlanner
from learning.model import MapEncoder, PoseEncoder, CnnGainEstimator, CNNGainEstimatorModel

import pickle
import math
import os
import numpy as np
import torch
import sys
import argparse
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import yaml

sys.path.append("../learning")
sys.path.append("../experiments")
sys.path.append('../simulator')
torch.set_num_threads(1)


def explore_worlds():
    cfg = Config()  # sets up defaults
    cfg.voxel_size = 0.2  # set custom params
    cfg.initial_safety_margin = 2

    # a little nicer plotting
    free = np.array([1, 1, 1, 1])
    occ = np.array([0, 0, 0, 1])
    unknown = np.array([0.85, 0.9, 0.95, 1])
    robot_cmp = ListedColormap(np.vstack((free, occ, unknown)))
    world_cmp = ListedColormap(np.vstack((free, occ)))

    if is_maze:
        # Maze
        cfg.world_type = cfg.world_type_maze
        cfg.mz_n_walls = [100, 200]
        cfg.mz_wall_strength = 0.5  # m
        cfg.mz_wall_length = [30, 45]  # m
        cfg.mz_wall_spacing = 5  # m
    else:
         # Cubes
        cfg.world_type = cfg.world_type_cubes_and_spheres
        cfg.cs_n_cubes = [10, 30]  # min, max
        cfg.cs_n_spheres = [10, 30]
        cfg.cs_cube_side_lengths = [1, 8]  # m
        cfg.cs_spheres_radii = [1, 4]  # m

    # setup the simulator
    sim = Simulator(cfg)

    # Execute planning steps.
    while True:
        plt.figure()
        sim.reset_world()
        sim.reset_robot()
        world_map = sim.get_world_map()
        x, y, yaw = sim.get_robot_pose()
        np.savetxt(world_file + '.txt', np.array((x, y, yaw)))
        plt.imshow(world_map, cmap=world_cmp)
        plt.imshow(world_map, cmap=world_cmp, interpolation='none')
        plt.arrow(y, x, math.sin(yaw) / cfg.voxel_size,
                  math.cos(yaw) / cfg.voxel_size, width=0.3 / cfg.voxel_size,
                  color='r')  # coordinate convention: x,y is in image coordinates, i.e. x = down, y = right

        # plt.show()
        plt.savefig(world_file + '.png')
        sim.save_state_to_file(world_file)
        observable = sim.get_explorable_area()
        print("Displaying the generated world map and initial robot pose.\n"
              "From this starting point, %.2f%s of the world (%i voxels) are observable."
              % (observable / np.shape(world_map)[0] / np.shape(world_map)[
            1] * 100,
                 '%', observable))

        # print("Enter '1' create a new world, or '2' to exit the demo:")
        # k = input()
        k = "2"
        if k == "2":
            break


def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def setup_sim(world_file, robot_pos=None):
    # Setup sim
    cfg = Config()
    cfg.local_submap_size_x = 50
    cfg.local_submap_size_y = 50
    sim = Simulator(cfg)
    sim.load_state_from_file(world_file)
    sim._cfg.voxel_size = 0.2  # set custom params
    sim._cfg.initial_safety_margin = 2

    if is_maze:
        # World Maze
        sim._cfg.world_type = Config.world_type_maze
        sim._cfg.mz_n_walls = [100, 200]
        sim._cfg.mz_wall_strength = 0.5  # m
        sim._cfg.mz_wall_length = [30, 45]  # m
        sim._cfg.mz_wall_spacing = 5  # m
        sim._cfg.local_submap_size_x = 50
        sim._cfg.local_submap_size_y = 50
    else:
        # World Cube and Sphere
        sim._cfg.world_type = Config.world_type_cubes_and_spheres
        sim._cfg.cs_n_cubes = [10, 30]  # min, max
        sim._cfg.cs_n_spheres = [10, 30]
        sim._cfg.cs_cube_side_lengths = [1, 8]  # m
        sim._cfg.cs_spheres_radii = [1, 4]  # m

    if robot_pos:
        sim.robot.load_initial_position(robot_pos)
    return sim


def run_planner(sim, n_samples, run_id, world_file, use_policy, policy_planner, local_planner, method='cvae', use_nbvp=False, localmap_size=50):
    # Args
    max_time = 20 * 60  # s
    d_time = []
    d_expl = []
    d_log = []
    d_comp = []
    d_distance = []
    d_global_plans = [] # None if was local, [reason, explored_after], 
    current_time = 0
    global total_voxels
    total_voxels = sim.get_explorable_area()

    is_locally_stuck_counter = 0
    while True:
        expl = sim.get_explored_area()
        print("Local Planning Step (t=%.1f, expl=%.1f%%)." % (
                current_time, expl / total_voxels * 100))
        d_time.append(current_time)
        d_expl.append(expl)
        global_plan = None
        if current_time > max_time:
            # Time limit reached.
            d_log.append("Time limit reached")
            break
        if is_locally_stuck_counter >= 20:
            global_plan = 'stuck'
        elif sim.is_in_local_minimum():
            # Local Minimum -> Global Planner
            global_plan = 'minimum'
        else:
            # Local Planning.
            if use_policy:
                plan, running_time, valid_point_found = policy_planner.plan()
                d_comp.append(running_time)
                if not valid_point_found:
                    global_plan = 'failed'
                else:
                    result = sim.move_to(plan[0], plan[1], plan[2])
            else:
                plan, running_time, valid_point_found = local_planner.plan()
                d_comp.append(running_time)
                if not valid_point_found:
                    global_plan = 'failed'
                else:
                    result = sim.move_to(plan.x, plan.y, plan.yaw)
                    if use_nbvp:
                        local_planner.update_tree()

        if global_plan is None:
            current_time = current_time + result.execution_time
            d_distance.append(result.path_length)
            if result.newly_observed_voxels == 0:
                is_locally_stuck_counter = is_locally_stuck_counter + 1
                max_time = max_time + result.execution_time
            else:
                is_locally_stuck_counter = 0
        else:
            plan = sim.call_global_planner(False)
            d_comp.append(np.nan)
            d_distance.append(np.nan)
            is_locally_stuck_counter = 0
            if not plan.success:
                d_log.append("Global Planning failed")
                break
            if use_nbvp:
                local_planner.reset()
        d_global_plans.append(global_plan)

    print("Finished exploration run")
    save({'time': d_time, 'expl': d_expl, 'comp': d_comp, 'log': d_log, 'global': d_global_plans, 'distance': d_distance},
         world_file +'/sample%i_run%i_' % (n_samples, run_id) + method)


def evaluate_multiple_planners(worlds, s, methods):
    dt = 15
    n_runs = 3
    continue_completed_runs = True

    max_steps = int(20 * 60 / dt) + 1
    means = {}
    std = {}
    for ii, method in enumerate(methods):
        local_data = np.zeros((max_steps, n_runs * len(worlds)))
        for j, world in enumerate(worlds):
            world_file = 'worlds/' + world
            sim = setup_sim(world_file)
            total_voxels = sim.get_explorable_area() 
            for n in range(n_runs):
                k = j * n_runs + n
                d = load(world_file +'/sample%i_run%i_' % (s, n) + method)
                time = d['time']
                expl = d['expl']

                t = 0
                current_index = 0
                finished = False
                # Interpolate the data
                for i in range(max_steps):
                    if not finished:
                        while time[current_index] <= t:
                            # Find the upper index
                            current_index = current_index + 1
                            if current_index >= len(time) - 1:
                                finished = True
                                break

                    if finished:
                        if continue_completed_runs:
                            local_data[i, k] = expl[-1]
                        else:
                            local_data[i, k] = np.nan
                    else:
                        local_data[i, k] = expl[current_index - 1] + (
                                    expl[current_index] - expl[
                                current_index - 1]) * (t - time[
                            current_index - 1]) / (time[current_index] - time[
                            current_index - 1])
                    t = t + dt
                local_data[:, k] /= total_voxels
        # Get mean and std for uniform planner and learnt planner
        means[ii] = np.nanmean(local_data, axis=1) * 100
        std[ii] = np.nanstd(local_data, axis=1) * 100
        print(np.shape(means))

    # Plotting
    plt.figure()
    colors = ['b', 'r', 'g', 'k', 'm', 'c', 'y']
    styles = ['-', '-.', ':', 'dashed', 'dashdot', '--', 'dotted']
    x = np.linspace(0, 20 * 60, max_steps) / 60
    for i in range(len(methods)):
        plt.plot(x, means[i], linestyle=styles[i], color=colors[i], label=methods[i])
        plt.fill_between(x, means[i] - std[i],
                         means[i] + std[i],
                         facecolor=colors[i], alpha=.2)

    plt.ylabel('Explored [%]')
    plt.xlabel('Exploration Time [min]')
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=x[-1])
    plt.legend()
    dir = 'evaluation/plots/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    name = dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_n_samples_" + str(s)
    plt.draw()
    plt.pause(2)
    plt.savefig(name + ".png")


def plot(worlds, n_samples, methods):
    for n_sample in n_samples:
        evaluate_multiple_planners(worlds, n_sample, methods)


def setup_local_planner(sim, n_sample, use_nbvp, localmap_size=50):
    # Args
    max_time = 20 * 60  # s
    sampling_steps = n_sample
    sampling_yaw = 1
    count_only_feasible_samples = True
    use_normalized_gain = True
    gain_decay_rate = 0.5 # Gain decay rate for NBVP
    maximum_edge_length = 1 # Max edge length for NBVP

    cfg = sim.config

    # Setup local planner
    action_bounds = [localmap_size / 2, localmap_size / 2,
                     math.pi]
    local_planner = None
    if use_nbvp:
        local_planner =LocalNBVPlanner(sim.robot, sampling_steps,action_bounds,gain_decay_rate, True, maximum_edge_length)
    else:
        local_planner = LocalSamplingPlanner(sim.robot, sampling_steps,
                                         sampling_yaw, action_bounds,
                                         count_only_feasible_samples,
                                         use_normalized_gain)
    return local_planner


def setup_policy(sim, local_planner, paths, method, n_samples):
    # Setup policy planner
    policy_planner = None
    if method == 'cvae':
        model = torch.load(paths['cvae'], map_location=torch.device('cpu'))
        model.to(torch.device("cpu"))
        model.eval()
        policy_planner = PolicyPlanner(sim, local_planner, model, method, n_samples)
    elif method == 'regression':
        regression_baseline = torch.load(paths['regression'], map_location=torch.device('cpu'))
        regression_baseline.to(torch.device('cpu'))
        regression_baseline.eval()
        policy_planner = PolicyPlanner(sim, local_planner, regression_baseline, method, n_samples)
    elif method == 'predict_gain':
        model_with_gain = torch.load(paths['gain_predict'], map_location=torch.device('cpu'))
        model_with_gain.to(torch.device("cpu"))
        model_with_gain.eval()
        policy_planner = PolicyPlanner(sim, local_planner, model_with_gain, method, n_samples)
    elif method == 'twostage_predict_gain' or method == 'uniform_predict':
        model = torch.load(paths['cvae'], map_location=torch.device('cpu'))
        model.to(torch.device("cpu"))
        model.eval()
        gain_estimator = torch.load(paths['gain_estimator'], map_location=torch.device('cpu'))
        gain_estimator.to(torch.device("cpu"))
        gain_estimator.eval()
        policy_planner = PolicyPlanner(sim, local_planner, model, method, n_samples, gain_estimator)
    elif method == 'cnn_gain_predict' or method == 'cnn_uniform_gain_predict':
        model = torch.load(paths['cvae'], map_location=torch.device('cpu'))
        model.to(torch.device("cpu"))
        model.eval()
        cnn_gain_estimator = torch.load(paths['cnn_gain_estimator'], map_location=torch.device('cpu'))
        cnn_gain_estimator.to(torch.device("cpu"))
        cnn_gain_estimator.eval()
        # from keras.models import load_model
        # map_encoder = load_model("map_encoder")
        # pose_encoder = load_model("pose_encoder")
        # gain_predictor = load_model("decoder")
        # cnn_gain_estimator = [map_encoder, pose_encoder, gain_predictor]
        policy_planner = PolicyPlanner(sim, local_planner, model, method, n_samples, cnn_gain_estimator)
    else:
        print("Policy Name Invalid")
    return policy_planner


def main():
    with open('config.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # global world file
    global is_maze
    is_maze = cfg['maze']

    # randomly generate maps
    if cfg['explore_world']:
        for file in cfg['worlds']:
            global world_file
            world_file = 'worlds/' + file
            if not os.path.isdir(world_file+'/'):
                os.mkdir(world_file+'/')
            explore_worlds()

    # evaluate planners
    n_runs = cfg['n_runs']
    n_samples = cfg['n_samples']
    if cfg['regression']: 
        n_samples = [1]
    
    for file in cfg['worlds']:
        print(file)
        world_file = 'worlds/' + file
        if not os.path.isdir(world_file+'/'):
            os.mkdir(world_file+'/')
        is_maze = cfg['maze']
        is_maze = False if file[0] == 'c' else True
        pos = np.loadtxt(world_file + '.txt')
        initial_position = [0,0,0]
        initial_position[0], initial_position[1], initial_position[2] = int(pos[0]), int(pos[1]), pos[2]
        print("evaluating map: ", file)
        for method in cfg['planners']:
            print("evaluating method: ", method)
            use_policy = False if method == "uniform" or method == 'nbvp' else True
            use_nbvp = True if method == 'nbvp' else False
            for run_id in range(n_runs):
                for n_sample in n_samples:
                    print("Running %i samples run %i" % (n_sample, run_id))
                    sim = setup_sim(world_file, initial_position)
                    local_planner = setup_local_planner(sim, n_sample, use_nbvp)
                    policy = setup_policy(sim, local_planner, cfg['path'], method, n_sample)
                    run_planner(sim, n_sample, run_id, world_file, use_policy, policy, local_planner, method, use_nbvp)

    # plot
    if cfg['plot']:
        plot(cfg['worlds'], cfg['n_samples'], cfg['planners'])


if __name__ == '__main__':
    main()
