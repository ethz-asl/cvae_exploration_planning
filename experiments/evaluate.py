import datetime
import pickle
import math
import os
import numpy as np
import torch
import yaml
import sys

from matplotlib import pyplot as plt

from cvae_exploration_planning.planning.baseline_nbvp import LocalNBVPlanner
from cvae_exploration_planning.simulator.config import Config
from cvae_exploration_planning.simulator.simulator import Simulator
from cvae_exploration_planning.planning.baseline_planner import LocalSamplingPlanner
from cvae_exploration_planning.planning.policy_planner import PolicyPlanner
# from cvae_exploration_planning.learning.model import CNNGainEstimatorModel, CVAE

sys.path.append("../learning")
# sys.path.append("../experiments")
# sys.path.append('../simulator')
torch.set_num_threads(1)


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
    sim._cfg.local_submap_size_x = 50
    sim._cfg.local_submap_size_y = 50

    if robot_pos:
        sim.robot.load_initial_position(robot_pos)
    return sim


def run_planner(sim, n_samples, run_id, output_dir, use_policy, policy_planner, local_planner, method='cvae', use_nbvp=False, time_limit=20):
    # Args
    max_time = time_limit * 60  # s
    d_time = []
    d_expl = []
    d_log = []
    d_comp = []
    d_distance = []
    d_global_plans = []  # None if was local, [reason, explored_after],
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
         output_dir + '/sample%i_run%i_' % (n_samples, run_id) + method)


def evaluate_multiple_planners(worlds, s, methods, output_dir, time_limit, n_runs):
    dt = 15
    continue_completed_runs = True
    max_steps = int(time_limit * 60 / dt) + 1

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
                d = load(
                    f"{output_dir}/{world}/sample{s if method != 'regression' else 1}_run{n}_{method}")
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
        # Get mean and std exploration
        means[ii] = np.nanmean(local_data, axis=1) * 100
        std[ii] = np.nanstd(local_data, axis=1) * 100

    # Plotting
    plt.figure()
    colors = ['b', 'r', 'g', 'k', 'm', 'c', 'y', 'orange', 'gold']
    styles = ['-', '-.', ':', '--'] * 3
    x = np.linspace(0, time_limit * 60, max_steps) / 60
    for i in range(len(methods)):
        plt.plot(x, means[i], linestyle=styles[i],
                 color=colors[i], label=methods[i])
        plt.fill_between(x, means[i] - std[i],
                         means[i] + std[i],
                         facecolor=colors[i], alpha=.2)

    plt.ylabel('Explored [%]')
    plt.xlabel('Exploration Time [min]')
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=x[-1])
    plt.legend()
    plt.draw()
    plt.pause(0.01)
    plt.savefig(f"{output_dir}/preformance_{s}_samples.png")


def setup_local_planner(sim, n_sample, use_nbvp, localmap_size=50):
    # Args
    sampling_steps = n_sample
    sampling_yaw = 1
    count_only_feasible_samples = True
    use_normalized_gain = True
    gain_decay_rate = 0.5  # Gain decay rate for NBVP
    maximum_edge_length = 1  # Max edge length for NBVP

    # Setup local planner
    action_bounds = [localmap_size / 2, localmap_size / 2,
                     math.pi]
    local_planner = None
    if use_nbvp:
        local_planner = LocalNBVPlanner(
            sim.robot, sampling_steps, action_bounds, gain_decay_rate, True, maximum_edge_length)
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
        policy_planner = PolicyPlanner(
            sim, local_planner, model, method, n_samples)
    elif method == 'regression':
        regression_baseline = torch.load(
            paths['regression'], map_location=torch.device('cpu'))
        regression_baseline.to(torch.device('cpu'))
        regression_baseline.eval()
        policy_planner = PolicyPlanner(
            sim, local_planner, regression_baseline, method, n_samples)
    elif method == 'cvae_joint':
        model_with_gain = torch.load(
            paths['cvae_joint'], map_location=torch.device('cpu'))
        model_with_gain.to(torch.device("cpu"))
        model_with_gain.eval()
        policy_planner = PolicyPlanner(
            sim, local_planner, model_with_gain, method, n_samples)
    elif method == 'twostage_predict_gain' or method == 'uniform_gain_predict':
        model = torch.load(paths['cvae'], map_location=torch.device('cpu'))
        model.to(torch.device("cpu"))
        model.eval()
        gain_estimator = torch.load(
            paths['gain_estimator'], map_location=torch.device('cpu'))
        gain_estimator.to(torch.device("cpu"))
        gain_estimator.eval()
        policy_planner = PolicyPlanner(
            sim, local_planner, model, method, n_samples, gain_estimator)
    elif method == 'cnn_gain_predict' or method == 'cnn_uniform_gain_predict':
        model = torch.load(paths['cvae'], map_location=torch.device('cpu'))
        model.to(torch.device("cpu"))
        model.eval()
        cnn_gain_estimator = torch.load(
            paths['cnn_gain_estimator'], map_location=torch.device('cpu'))
        cnn_gain_estimator.to(torch.device("cpu"))
        cnn_gain_estimator.eval()
        policy_planner = PolicyPlanner(
            sim, local_planner, model, method, n_samples, cnn_gain_estimator)
    else:
        print(f"Policy name '{method}' is invalid!")
    return policy_planner


def main():
    with open('config.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Setup files and dirs.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/{timestamp}"
    os.makedirs(output_dir)

    # Run all experiments.
    for world in cfg['worlds']:
        print(f"===== Running on world '{world}' =====")
        world_file = 'worlds/' + world
        os.mkdir(os.path.join(output_dir, world))
        if not os.path.isfile(world_file + '.p'):
            print(f"Specified world file '{world_file}.p' does not exist.")
            continue
        if not os.path.isfile(world_file + '.txt'):
            print(f"Specified world file '{world_file}.txt' does not exist.")
            continue

        # Setup the simulation.
        pos = np.loadtxt(world_file + '.txt')
        initial_position = [int(pos[0]), int(pos[1]), pos[2]]
        for method in cfg['planners']:
            n_samples = cfg['n_samples']
            if method == "regression":
                n_samples = [1]
            print(f"----- Evaluating method '{method}' -----")
            use_policy = False if method == "uniform" or method == 'nbvp' else True
            use_nbvp = True if method == 'nbvp' else False
            for run_id in range(cfg['n_runs']):
                for n_sample in n_samples:
                    print("Running %i samples, run number %i/%i." %
                          (n_sample, run_id + 1, cfg['n_runs']))
                    sim = setup_sim(world_file, initial_position)
                    local_planner = setup_local_planner(
                        sim, n_sample, use_nbvp)
                    policy = setup_policy(
                        sim, local_planner, cfg['path'], method, n_sample) if use_policy else None
                    run_planner(sim, n_sample, run_id, output_dir + '/' + world, use_policy,
                                policy, local_planner, method, use_nbvp, cfg['time_limit'])

    # Plotting of results
    if cfg['plot']:
        print("Plotting results.")
        for n_sample in cfg['n_samples']:
            evaluate_multiple_planners(
                cfg['worlds'], n_sample, cfg['planners'], output_dir, cfg['time_limit'], cfg['n_runs'])

    print("Evaluation compelted.")


if __name__ == '__main__':
    main()
