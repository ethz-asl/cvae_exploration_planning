from distutils.command.config import config
import numpy as np
import torch
import math


# x_dim = 3
# local_img_size = 100

def criteria(recon_batch, init_batch):
    """ Compute loss function for position and orientation. """
    loss_xy = torch.nn.MSELoss(recon_batch[:, 0:2], init_batch[:, 0:2])
    yaw_diff = torch.abs(recon_batch[:, 2] - init_batch[:, 2]) # suppose it is within (0,2pi)
    yaw_diff_adjusted = torch.min(yaw_diff, np.pi*2-yaw_diff)
    loss_yaw = torch.mean(yaw_diff_adjusted**2)
    return loss_yaw + loss_xy

def kl_regularizer(mu, logvar, kl_weight):
    """ KL Divergence regularizer """
    # it still returns a vector with dim: (batchsize,)
    return kl_weight * 2 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1)

# one-hot encoder the data
def one_hot_encoder(data, x_dim, local_img_size):
    new_data = []
    for i in range(len(data)):
        sample = data[i]
        new_sample = np.zeros(x_dim + local_img_size * 2)
        encoder1 = np.zeros(local_img_size)
        encoder2 = np.zeros(local_img_size)
        for j in range(local_img_size):
            if sample[j+x_dim] == 0:  # free
                encoder1[j] = 1
            if sample[j+x_dim] == 1:  # occupied
                encoder2[j] = 1
            # otherwise: unobserved
        new_sample[0:x_dim] = sample[0:x_dim]
        new_sample[x_dim:x_dim + local_img_size] = encoder1
        new_sample[x_dim + local_img_size:x_dim+local_img_size*2] = encoder2
        new_data.append(new_sample)
    return new_data


def one_hot_encoder_map(data, local_img_size):
    new_data = []
    for i in range(len(data)):
        sample = data[i]
        new_sample = np.zeros(local_img_size * 2)
        encoder1 = np.zeros(local_img_size)
        encoder2 = np.zeros(local_img_size)
        for j in range(local_img_size):
            if sample[j] == 0:  # free
                encoder1[j] = 1
            if sample[j] == 1:  # occupied
                encoder2[j] = 1
            # otherwise: unobserved
        new_sample[0:local_img_size] = encoder1
        new_sample[local_img_size:local_img_size * 2] = encoder2
        new_data.append(new_sample)
    return new_data


def coord_transform(x_in, y_in, yaw_in):
        """ 
        converts input x,y,yaw to polar relative coords.
        Returns r in [0, sqrt(2)/2*LocalMapSize], theta, psi in [-pi, pi]
        """
        r = (x_in ** 2 + y_in ** 2) ** 0.5
        theta = np.arctan2(y_in, x_in)
        psi = yaw_in - theta
        if psi >= 2.0 * math.pi:
            psi = psi - 2.0 * math.pi
        return r, theta, psi