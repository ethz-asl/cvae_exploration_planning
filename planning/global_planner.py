import numpy as np
import random
import math
from scipy import ndimage

from simulator.config import Config
from simulator.robot import Robot
from planning.rrt_star import RRTStar


class GlobalPlanner:
    def __init__(self, cfg: Config, robot: Robot):
        self.cfg = cfg
        self._robot = robot

    class PlanningResult:
        """ Contains the output action and statistics of the planner """

        def __init__(self):
            self.success = False
            self.x = 0  # map coords
            self.y = 0
            self.yaw = 0
            self.path = []  # map coords
            self.checked_frontier_points = 0
            self.path_length = 0  # m
            self.path_execution_time = 0  # s

    def compute_frontiers(self, observed_map=None, initial_pose=None):
        """ Compute all frontier points in the given map. Initial position is [x, y].
        Outputs the frontiers as image labeled by frontier cluster.
         """
        # By default use the robot state for observed_map and start pose.
        if observed_map is None:
            observed_map = self._robot.observed_map
        if initial_pose is None:
            initial_pose = [self._robot.position_x, self._robot.position_y]
        mask = observed_map == 0
        labeled_image, _ = ndimage.measurements.label(mask)
        mask = labeled_image == labeled_image[
            int(initial_pose[0]), int(initial_pose[1])]  # reachable
        frontier_mask = ndimage.binary_dilation(mask)
        frontiers = (observed_map == 2) & frontier_mask
        labeled_frontiers, num_frontiers = ndimage.measurements.label(frontiers,
                                                                      structure=[
                                                                          [1, 1,
                                                                           1],
                                                                          [1, 1,
                                                                           1],
                                                                          [1, 1,
                                                                           1]])
        # remove too small frontiers
        for i in range(num_frontiers):
            frontier_img = labeled_frontiers == i + 1
            if np.sum(frontier_img) < self.cfg.glob_min_frontier_size:
                frontiers[frontier_img] = 0
        return frontiers

    def is_local_minimum(self, observed_map=None, initial_pose=None):
        """ Check whether there any observable frontiers in the given map. """
        return np.max(self.compute_frontiers(observed_map, initial_pose)) == 0

    def plan(self, verify_path=False):
        """ Compute a closest frontier if there is still one left in the map.
        Check for reachability with RRT* algorithm.
        Initial position is [x, y].
        verify_path: False to get the frontier as goal only,
                    True to compute a feasible path.
         """
        result = GlobalPlanner.PlanningResult()

        # Compute frontiers
        frontiers = self.compute_frontiers()
        labeled_frontiers, num_frontiers = ndimage.measurements.label(frontiers,
                                                                      structure=[
                                                                          [1, 1,
                                                                           1],
                                                                          [1, 1,
                                                                           1],
                                                                          [1, 1,
                                                                           1]])
        # remove too small frontiers
        for i in range(num_frontiers):
            frontier_img = labeled_frontiers == i + 1
            if np.sum(frontier_img) < self.cfg.glob_min_frontier_size:
                frontiers[frontier_img] = 0

        # Convert to frontiers points
        xx, yy = np.mgrid[:np.shape(frontiers)[0], :np.shape(frontiers)[1]]
        x_frontier = list(xx[frontiers])
        y_frontier = list(yy[frontiers])

        # find best reachable frontier
        while True:
            if not x_frontier:
                # No reachable frontiers left in the map
                return result
            goal_index = self.choose_goal(x_frontier, y_frontier,
                                          self._robot.position_x,
                                          self._robot.position_y)

            # Set the target yaw
            yaw = self.cfg.initial_yaw
            if yaw is None:
                yaw = random.uniform(0, math.pi * 2)
            goal = [x_frontier.pop(goal_index), y_frontier.pop(goal_index), yaw]


            # Ignore too close by samples.
            j=0
            for i in range(len(x_frontier)):
                dist = (((goal[0] - x_frontier[j]) ** 2.0 + (
                            goal[1] - y_frontier[j]) ** 2.0) ** 0.5) * self.cfg.voxel_size
                if dist < self.cfg.glob_frontier_consolidation_radius:
                    x_frontier.pop(j)
                    y_frontier.pop(j)
                else:
                    j = j + 1

            # Sample a goal point in free space.
            max_samples = 1000  # worst case try another frontier.
            goal_is_valid = False
            sampling_range = int(self.cfg.glob_radius / self.cfg.voxel_size) + 1
            for i in range(max_samples):
                # Also check visibility.
                x = goal[0] + random.randint(-sampling_range, sampling_range)
                y = goal[1] + random.randint(-sampling_range, sampling_range)
                if x < 0 or x >= np.shape(self._robot.observed_map)[
                    0] or y < 0 or y >= np.shape(self._robot.observed_map)[
                    1]:
                    # Out of bounds
                    continue
                elif self._robot.observed_map[x, y] != 0:
                    # Not in free space
                    continue
                else:
                    goal[0] = x
                    goal[1] = y
                    goal_is_valid = True
                    break

            if not goal_is_valid:
                continue
            if not verify_path:
                # Only needed a goal point a no path.
                result.success = True
                result.x = goal[0]
                result.y = goal[1]
                result.yaw = goal[2]
                return result

            # Check whether the goal is reachable and compute a path.
            rrt_star = RRTStar(self._robot.observed_map,
                               [self._robot.position_x, self._robot.position_y],
                               goal[:2],
                               self.cfg.rrt_expand_dis / self.cfg.voxel_size,
                               1.0, self.cfg.rrt_goal_sample_rate,
                               self.cfg.rrt_max_iter, self.cfg.rrt_min_iter,
                               self.cfg.rrt_connect_dist / self.cfg.voxel_size)
            path = rrt_star.plan()
            result.checked_frontier_points = result.checked_frontier_points + 1
            if not path:
                # Failed to find a path
                continue

            # Found a path
            result.success = True
            result.path = path
            result.x = goal[0]
            result.y = goal[1]
            result.yaw = goal[2]
            return self.evaluate_path_cost(result)

    def choose_goal(self, x_frontier, y_frontier, x_start, y_start):
        """ Select a goal point and pop it from the candidates.
        x_frontier: list of x coords of all frontiers
        y_frontier: list of y coords of all frontiers
        x/y_start: initial position of the robot
        """
        if len(x_frontier) == 0:
            return []
        # Choose goal frontier according to policy.
        i = 0
        if self.cfg.glob_frontier_selection == Config.glob_frontier_selection_closest:  # find closest frontier
            dists = (np.array(x_frontier) - x_start) ** 2 + (
                    np.array(y_frontier) - y_start) ** 2
            i = np.where(dists == np.amin(dists))[0][0]
        else:  # find random frontier
            i = random.randint(0, len(x_frontier) - 1)
        return i

    def evaluate_path_cost(self, planning_result: PlanningResult):
        """ Compute the length and execution time of a global path """
        planning_result.path_length = 0
        planning_result.path_execution_time = 0
        if planning_result.path:
            for i in range(1, len(planning_result.path)):
                dist = ((planning_result.path[i][0] -
                         planning_result.path[i - 1][0]) ** 2 + (
                                planning_result.path[i][1] -
                                planning_result.path[i - 1][1]) ** 2) ** 0.5
                planning_result.path_length = planning_result.path_length \
                                              + dist * self.cfg.voxel_size
                # Only consider the position velocity for global plans
                self._robot.pos_ramp.setup_from_goal(dist)
                planning_result.path_execution_time = planning_result.path_execution_time + self._robot.pos_ramp.t_max
        return planning_result
