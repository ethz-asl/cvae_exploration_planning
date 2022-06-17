import time

from simulator.robot import Robot
import random
import numpy as np
import math

""" This file contains some baseline sampling based planners to compare against """


class LocalSamplingPlanner:
    """ A planner that samples n local viewpoints and then selects the best one """

    def __init__(self, robot: Robot, n_samples, n_yaw, action_bounds, only_count_feasible_samples=False, use_normalized_gain=False):
        self._robot = robot
        self.n_samples = n_samples  # Fixed number of samples evaluated at every iteration
        self.n_yaw = n_yaw  # Number of yaw angles to sample
        self.only_count_feasible_samples = only_count_feasible_samples
        self.use_normalized_gain = use_normalized_gain
        self.action_bounds = action_bounds

    class PlanningResult:
        """ This class represents the output action and statistics of the planner """

        def __init__(self):
            self.x = 0
            self.y = 0
            self.yaw = 0
            self.feasible_samples = 0
            self.total_samples = 0
            self.gain = 0

    def plan(self):
        """ Returns the planning result of this planner """
        time_start = time.time()
        valid_point_found = False
        counter = 0
        try_counter = 0
        feasible_counter = 0
        best_move = []  # Stores the feasible moves to select best one
        best_gain = 0
        while not valid_point_found or counter < self.n_samples:
            try_counter = try_counter + 1
            # Sample positions
            x = random.randint(-self.action_bounds[0], self.action_bounds[0])
            y = random.randint(-self.action_bounds[1], self.action_bounds[1])
            _, is_safe = self._robot.check_move_feasible(x, y)
            if is_safe:
                feasible_counter = feasible_counter + 1
                # Sample n_yaw different yaws for every feasible point
                for i in range(self.n_yaw):
                    yaw = (i + np.random.random()) * 2. * math.pi / self.n_yaw
                    new_voxels = self.get_number_of_visible_voxels(x, y, yaw)
                    gain = self.compute_gain(x, y, yaw, new_voxels)
                    if not valid_point_found or gain > best_gain:
                        # First found point
                        best_gain = gain
                        best_move = [x, y, yaw]
                        valid_point_found = True
            if not self.only_count_feasible_samples or is_safe:
                counter = counter + 1
            if try_counter == 10000 and not valid_point_found:
                best_move = [0, 0, 0]
                break
        result = LocalSamplingPlanner.PlanningResult()
        result.x = best_move[0]
        result.y = best_move[1]
        result.yaw = best_move[2]
        result.feasible_samples = feasible_counter
        result.total_samples = try_counter
        result.gain = best_gain
        time_end = time.time()
        running_time = time_end - time_start
        return result, running_time, valid_point_found

    def get_number_of_visible_voxels(self, x, y, yaw):
        """ Does naive ray-casting to identify expected visible voxels based on
         the robot map, similar to robot.take_sensor_measurement.
         x,y, yaw are in robot frame. """
        x = self._robot.position_x + x
        y = self._robot.position_y + y
        n_rays = self._robot.cfg.camera_range * math.pi * 2 * self._robot.cfg.camera_fov / 360 / self._robot.cfg.voxel_size
        n_rays = math.ceil(n_rays * 1.5)  # make sure we get full coverage of voxels,
        dphi = self._robot.cfg.camera_fov * math.pi / 180 / (n_rays - 1)
        visible_map = np.full(np.shape(self._robot.observed_map), -1)
        for i in range(n_rays):
            r = 0
            phi = yaw - self._robot.cfg.camera_fov * math.pi / 180 / 2 + i * dphi
            while r <= self._robot.cfg.camera_range / self._robot.cfg.voxel_size:
                r = r + 1
                x_ray, y_ray = self._robot.bound_query_voxels(
                    x + r * math.cos(phi),
                    y + r * math.sin(phi))
                visible_map[x_ray, y_ray] = self._robot.observed_map[x_ray, y_ray]
                if visible_map[x_ray, y_ray] == 1:
                    break
        # Count the number of expected visible voxels that are unknown
        return np.sum(visible_map == 2)

    def compute_gain(self, x, y, yaw, new_voxels):
        """ Compute a total value for a viewpoint. """
        if self.use_normalized_gain:
            # Normalized gain: number of voxels divided by time
            distance = (float(x) ** 2 + float(y) ** 2) ** 0.5
            yaw_diff = math.fmod(yaw - self._robot.yaw, math.pi * 2)
            if math.fabs(yaw_diff) > math.pi:
                yaw_diff = yaw_diff - np.sign(yaw_diff) * math.pi * 2
            self._robot.pos_ramp.setup_from_goal(distance)
            self._robot.yaw_ramp.setup_from_goal(math.fabs(yaw_diff))
            max_time = max(self._robot.pos_ramp.t_max, self._robot.yaw_ramp.t_max)
            return new_voxels / max_time
        else:
            # Simple implementation: just the number of new voxels
            return new_voxels
