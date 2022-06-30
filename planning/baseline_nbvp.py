import time
import random
import numpy as np
import math

from cvae_exploration_planning.simulator.robot import Robot
from cvae_exploration_planning.planning.baseline_planner import LocalSamplingPlanner

""" 
This file contains an implementation of the RH-NBVP [1] as baseline to compare against.

[1] Bircher, A., Kamel, M., Alexis, K., Oleynikova, H., & Siegwart, R. (2018). Receding horizon path planning for 3D exploration and surface inspection. Autonomous Robots, 42(2), 291-306.
 """


class LocalNBVPlanner(LocalSamplingPlanner):
    """ A planner that builds an RRT of viewpoints and selects the first point of the best branch. """

    def __init__(self, robot: Robot, n_samples, action_bounds, gain_decay_rate, only_count_feasible_samples=False, max_edge_length=np.inf, verbose=False):
        LocalSamplingPlanner.__init__(
            self, robot, n_samples, 0, action_bounds, only_count_feasible_samples, False)
        # Params.
        self.gain_decay_rate = gain_decay_rate  # Gain decay rate \lambda
        # Max connection distance in px, input is in m.
        self.max_edge_length = max_edge_length / robot.cfg.voxel_size
        self.verbose = verbose

        # State.
        # The root node of the tree. Nodes are in global frame.
        self.root = None
        # Cached information about which node to execute next.
        self.next_node = None
        self.previous_global_pos_x = None
        self.previous_global_pos_y = None

        # Setup.
        self.reset()

    class Node:
        def __init__(self, x=0, y=0, yaw=0, gain=0) -> None:
            self.x = x
            self.y = y
            self.yaw = yaw
            self.gain = gain
            self.children = []
            self.parent = None

        def num_nodes(self) -> int:
            """ Compute the number of children in the subtree. """
            num_nodes = 1
            for child in self.children:
                num_nodes += child.num_nodes()
            return num_nodes

    def reset(self):
        """ Resets the planner's tree to start exploration again. """
        self.previous_global_pos_x = self._robot.position_x
        self.previous_global_pos_y = self._robot.position_y
        self.root = LocalNBVPlanner.Node(
            self.previous_global_pos_x, self.previous_global_pos_y, self._robot.yaw)
        self.next_node = None

    def plan(self):
        """ Returns the planning result of this planner """
        time_start = time.time()
        valid_point_found = False
        counter = 0
        try_counter = 0
        feasible_counter = 0
        num_samples_to_add = self.n_samples + 1 - self.root.num_nodes()
        if num_samples_to_add > 0:
            while not valid_point_found or counter < num_samples_to_add:
                try_counter = try_counter + 1
                sample_success = self.add_sample()
                if sample_success:
                    valid_point_found = True
                    feasible_counter = feasible_counter + 1
                if sample_success or not self.only_count_feasible_samples:
                    counter = counter + 1
                if try_counter == 10000 and not valid_point_found:
                    break
        if self.verbose:
            print(
                f"Samples: {try_counter} tries, {feasible_counter} feasible, {num_samples_to_add} to add, {self.root.num_nodes()} tree size.")
        best_node, best_gain = self.get_best_node()
        result = LocalSamplingPlanner.PlanningResult()
        # convert to local frame.
        result.x = best_node.x - self.previous_global_pos_x
        result.y = best_node.y - self.previous_global_pos_y
        result.yaw = best_node.yaw
        result.feasible_samples = feasible_counter
        result.total_samples = try_counter
        result.gain = best_gain
        time_end = time.time()
        running_time = time_end - time_start
        self.next_node = best_node
        return result, running_time, valid_point_found

    def add_sample(self) -> bool:
        # Sample positions
        x = self.previous_global_pos_x + random.randint(-self.action_bounds[0],
                                                        self.action_bounds[0])
        y = self.previous_global_pos_y + random.randint(-self.action_bounds[1],
                                                        self.action_bounds[1])
        # Find closest node.
        closest_node = [None]   # Pass as list s.t. reference is mutable.

        def recursive_search(node: LocalNBVPlanner.Node, closest_node: list, shortest_dist_squared: list) -> None:
            dist_squared = (x - node.x) ** 2 + (y-node.y)**2
            if dist_squared < shortest_dist_squared[0]:
                shortest_dist_squared[0] = dist_squared
                closest_node[0] = node
            for child in node.children:
                recursive_search(child, closest_node, shortest_dist_squared)
        recursive_search(self.root, closest_node, [np.inf])
        closest_node = closest_node[0]

        distance = ((closest_node.x-x) ** 2 +
                    (closest_node.y - y) ** 2) ** 0.5
        if distance > self.max_edge_length:
            return False

        _, is_safe = self._robot.check_move_feasible(
            x-self.previous_global_pos_x, y-self.previous_global_pos_y, None, True, closest_node.x-self.previous_global_pos_x, closest_node.y-self.previous_global_pos_y)
        if not is_safe:
            return False
        # Add the node to the tree.
        new_node = LocalNBVPlanner.Node(
            x, y, np.random.random() * 2. * math.pi)
        new_node.parent = closest_node
        self.evalute_node_gain(new_node)
        closest_node.children.append(new_node)
        return True

    def evalute_node_gain(self, node: Node) -> None:
        if node.parent is None:
            node.gain = 0
            return
        new_voxels = self.get_number_of_visible_voxels(
            node.x-self.previous_global_pos_x, node.y-self.previous_global_pos_y, node.yaw)

        distance = ((node.parent.x - node.x) ** 2 +
                    (node.parent.y - node.y) ** 2) ** 0.5

        node.gain = node.parent.gain + new_voxels * \
            math.exp(-self.gain_decay_rate * distance)

    def get_best_node(self):
        """ Find the next node that has the highest gain in it's subtree. Return the node and the highest gain. """
        if self.root.children:
            highest_gain = -1
            best_node = None

            def recursive_gain_search(node: LocalNBVPlanner.Node, highest_gain) -> float:
                highest_gain = np.maximum(node.gain, highest_gain)
                for child in node.children:
                    highest_gain = np.maximum(recursive_gain_search(
                        child, highest_gain), highest_gain)
                return highest_gain
            for child in self.root.children:
                gain = recursive_gain_search(child, 0)
                if gain > highest_gain:
                    highest_gain = gain
                    best_node = child
            return best_node, highest_gain
        else:
            if self.verbose:
                print("No next step found: only the root is in the tree.")
            return self.root, 0

    def update_tree(self) -> None:
        """ Update the tree after the previously determined best path was executed. This assumes that the simulation executed the porposed best path. """
        if self.next_node is None:
            # Initialization. (Just reset for safety, should already be reset.)
            self.reset()
            if self.verbose:
                print("Reset the tree for initialization.")
            
        # Check path was executed correctly.
        elif self.next_node.x != self._robot.position_x or self.next_node.y != self._robot.position_y:
            if self.verbose:
                print(
                    f"Previous plan was not executed ({self.next_node.x}, {self.next_node.y} vs {self._robot.position_x}, {self._robot.position_y}), resetting the tree.")
            self.reset()
        else:
            # Update tracking
            self.previous_global_pos_x = self._robot.position_x
            self.previous_global_pos_y = self._robot.position_y

            # Move the root one node forward.
            num_previous_nodes = self.root.num_nodes()
            self.root = self.next_node
            self.root.parent = None

            def recursive_update(node: LocalNBVPlanner.Node):
                self.evalute_node_gain(node)
                for child in node.children:
                    recursive_update(child)

            recursive_update(self.root)
            if self.verbose:
                print(
                    f"Updated the tree: {num_previous_nodes}->{self.root.num_nodes()} nodes.")
