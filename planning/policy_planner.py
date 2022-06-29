import math
import time
import random
import numpy as np
import torch
from simulator.simulator import Simulator
from learning.util import coord_transform, one_hot_encoder_map


class PolicyPlanner:
    """ policy planner that samples from the learned distribution. """

    def __init__(self, sim: Simulator, local_planner, model, method='cvae', n_sample=10, gain_estimator=None):
        self._sim = sim
        self._localPlanner = local_planner
        self._method = method
        self.n_sample = n_sample
        self.model = model
        self.maximum_try = 1000
        self.device = 'cpu'
        self.dtype = torch.float32
        self.action_bounds = local_planner.action_bounds
        self.n_yaw = local_planner.n_yaw
        self.pooling_stride = 5
        self.dim_latent = 3
        self.local_img_size = 100
        if self._method == 'twostage_predict_gain' or self._method == 'uniform_predict':
            self.gain_estimator = gain_estimator
        if self._method == 'cnn_uniform_gain_predict' or self._method == 'cnn_gain_predict':
            self.gain_predictor = gain_estimator

    def get_local_condition(self, n_stack=1):
        local_map = self._sim.get_robot_local_submap()
        local_map_zip = []
        for ii in range(10):
            for jj in range(10):
                block = local_map[self.pooling_stride * ii: self.pooling_stride * ii + self.pooling_stride,
                                  self.pooling_stride * jj: self.pooling_stride * jj + self.pooling_stride]
                if 1 in block:
                    local_map_zip.append(1)
                elif 2 in block:
                    local_map_zip.append(2)
                else:
                    local_map_zip.append(0)
        local_map_zip = np.array(local_map_zip)
        local_map_zip = one_hot_encoder_map(
            np.repeat([local_map_zip], n_stack, axis=0), self.local_img_size)
        cond = torch.tensor(
            local_map_zip, dtype=self.dtype, device=self.device)
        return cond

    def localmap_downsample(self, ifonehot=False, zipsize=25):
        localmap = self._sim.get_robot_local_submap()
        mapsize = len(localmap)
        ratio = int(mapsize/zipsize)
        if(ratio <= 1):
            print("error in downsampling localmap")
            return
        localMap_zip = []
        for ii in range(zipsize):
            for jj in range(zipsize):
                block = localmap[ratio*ii: ratio *
                                 (ii+1), ratio*jj: ratio*(jj+1)]
                if (1 in block):  # obstacle
                    localMap_zip.append(1)
                elif (2 in block):  # unobserved
                    localMap_zip.append(2)
                else:  # free
                    localMap_zip.append(0)
        localMap_zip = np.array(localMap_zip)  # (localMap_zip 100*1)
        # localMap_zip_2D = localMap_zip.reshape(zipsize, zipsize)  # (localMap_zip_2D 10*10)
        if ifonehot:
            l = len(localMap_zip)
            localMap_onehot = np.zeros(2*l)
            for k in range(len(localMap_zip)):
                if localMap_zip[k] == 0:  # free
                    localMap_onehot[k] = 1
                if localMap_zip[k] == 1:  # occupied
                    localMap_onehot[k+l] = 1
                # otherwise: unobserved
            return localMap_onehot
        return localMap_zip

    def plan(self):
        best_gain = -np.inf
        best_move = [0, 0, 0]
        valid_point_found = False
        counter = 0
        try_counter = 0

        ''' cvae: evaluating the real gain for each sample'''
        if self._method == 'cvae':
            time_start = time.time()
            cond = self.get_local_condition(1)

            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                z = torch.randn(1, self.dim_latent)
                z_in = torch.cat((z, cond), 1)
                result = self.model.decoder(z_in)
                result = result.detach().numpy().astype("float64")[0]
                _, is_safe = self._sim.robot.check_move_feasible(
                    result[0], result[1])
                if is_safe:
                    new_voxels = self._localPlanner.get_number_of_visible_voxels(
                        result[0], result[1], result[2])
                    gain = self._localPlanner.compute_gain(
                        result[0], result[1], result[2], new_voxels)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = [result[0], result[1], result[2]]
                        valid_point_found = True
                    counter = counter + 1
                # after n_sample tries, still not found a valid point, policy fails
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        # return best_move, ask_status, time.time() - time_start, resorting_time, valid_point_found, try_counter

        ''' Predict the gain together with actions, not evaluate real gain '''
        if self._method == 'predict_gain':
            time_start = time.time()
            cond = self.get_local_condition(1)
            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                z = torch.randn(1, self.dim_latent)
                z_in = torch.cat((z, cond), 1)
                result = self.model.decoder(z_in)
                result = result.detach().numpy().astype("float64")[0]
                _, is_safe = self._sim.robot.check_move_feasible(
                    result[0], result[1])
                if is_safe:
                    new_voxels = result[3]
                    gain = self._localPlanner.compute_gain(
                        result[0], result[1], result[2], new_voxels)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = [result[0], result[1], result[2]]
                        valid_point_found = True
                    counter = counter + 1
                # after n_sample tries, still not found a valid point, policy fails
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        ''' regression baseline '''
        if self._method == 'regression':
            time_start = time.time()
            cond = self.get_local_condition(1)
            result = self.model(cond)
            result = result.detach().numpy().astype("float64").reshape(-1)
            i = 0
            count = 0
            _, is_safe = self._sim.robot.check_move_feasible(
                result[0], result[1])
            if is_safe:
                count = count + 1
                best_move = [result[0], result[1], result[2]]
                valid_point_found = True
            try_counter += 1

        ''' a seperate gain estimator to predict gain of cvae samples, not evaluate real gain '''
        if self._method == 'twostage_predict_gain':
            time_start = time.time()
            cond = self.get_local_condition(1)
            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                z = torch.randn(1, self.dim_latent)
                z_in = torch.cat((z, cond), 1)
                result = self.model.decoder(z_in)
                result_numpy = result.detach().numpy().astype("float64")[0]
                transferred_result = torch.tensor(np.array(coord_transform(
                    result_numpy[0], result_numpy[1], result_numpy[2])), dtype=torch.float).reshape(1, -1)
                _, is_safe = self._sim.robot.check_move_feasible(
                    result_numpy[0], result_numpy[1])
                if is_safe:
                    new_voxels = self.gain_estimator(transferred_result, cond)
                    gain = self._localPlanner.compute_gain(
                        result_numpy[0], result_numpy[1], result_numpy[2], new_voxels)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = [result_numpy[0],
                                     result_numpy[1], result_numpy[2]]
                        valid_point_found = True
                    counter = counter + 1
                # after n_sample tries, still not found a valid point, policy fails
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        ''' ranomly sampling and use the gain predictor to estimate the gain'''
        if self._method == 'uniform_predict':
            time_start = time.time()
            cond = self.get_local_condition(1)
            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                # Sample positions
                x = random.randint(-self.action_bounds[0],
                                   self.action_bounds[0])
                y = random.randint(-self.action_bounds[1],
                                   self.action_bounds[1])
                _, is_safe = self._sim.robot.check_move_feasible(x, y)
                if is_safe:
                    # Sample n_yaw different yaws for every feasible point
                    for i in range(self.n_yaw):
                        yaw = (i + np.random.random()) * \
                            2. * math.pi / self.n_yaw
                        new_voxels = self.gain_estimator(torch.tensor(
                            np.array(coord_transform(x, y, yaw)), dtype=torch.float).reshape(1, -1), cond)
                        gain = self._localPlanner.compute_gain(
                            x, y, yaw, new_voxels)
                        if not valid_point_found or gain > best_gain:
                            # First found point
                            best_gain = gain
                            best_move = [x, y, yaw]
                            valid_point_found = True
                    counter = counter + 1
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        ''' use cnn as the gain estimator '''
        if self._method == "cnn_gain_predict":
            time_start = time.time()
            cond_z = self.get_local_condition(1)
            cond = self.localmap_downsample()
            cond = np.array(cond).reshape(1, 1, 25, -1)
            # map_embed = self.map_encoder(cond)
            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                z = torch.randn(1, self.dim_latent)
                z_in = torch.cat((z, cond_z), 1)
                result = self.model.decoder(z_in)
                result_numpy = result.detach().numpy().astype("float64")[0]
                result4cnn = np.zeros(4)
                result4cnn[0:2] = result_numpy[0:2]
                result4cnn[2] = math.cos(result_numpy[2])
                result4cnn[3] = math.sin(result_numpy[2])
                result4cnn = result4cnn.reshape(-1, 4)
                _, is_safe = self._sim.robot.check_move_feasible(
                    result_numpy[0], result_numpy[1])
                if is_safe:
                    # pose_embed = self.pose_encoder(result4cnn)
                    new_voxels = float(self.gain_predictor(torch.FloatTensor(
                        cond), torch.FloatTensor(result4cnn)).reshape(-1))
                    gain = self._localPlanner.compute_gain(
                        result_numpy[0], result_numpy[1], result_numpy[2], new_voxels)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = [result_numpy[0],
                                     result_numpy[1], result_numpy[2]]
                        valid_point_found = True
                    counter = counter + 1
                # after n_sample tries, still not found a valid point, policy fails
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        ''' use cnn as the gain estimator and the uniform samples '''
        if self._method == "cnn_uniform_gain_predict":
            time_start = time.time()
            cond = self.localmap_downsample()
            cond = np.array(cond).reshape(1, 1, 25, -1)
            while not valid_point_found or counter < self.n_sample:
                try_counter = try_counter + 1
                # Sample positions
                x = random.randint(-self.action_bounds[0],
                                   self.action_bounds[0])
                y = random.randint(-self.action_bounds[1],
                                   self.action_bounds[1])
                _, is_safe = self._sim.robot.check_move_feasible(x, y)
                if is_safe:
                    # Sample n_yaw different yaws for every feasible point
                    for i in range(self.n_yaw):
                        yaw = (i + np.random.random()) * \
                            2. * math.pi / self.n_yaw
                        new_voxels = float(self.gain_predictor(torch.FloatTensor(cond), torch.FloatTensor(
                            np.array([x, y, math.cos(yaw), math.sin(yaw)]).reshape(-1, 4))).reshape(-1))
                        gain = self._localPlanner.compute_gain(
                            x, y, yaw, new_voxels)
                        if not valid_point_found or gain > best_gain:
                            # First found point
                            best_gain = gain
                            best_move = [x, y, yaw]
                            valid_point_found = True
                    counter = counter + 1
                if try_counter == self.maximum_try and not valid_point_found:
                    break

        return best_move, time.time() - time_start, valid_point_found
