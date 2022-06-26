import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import datetime
import os
from data import Dataset
from model import VAE, Encoder, Decoder, CVAE
import argparse
from torch.multiprocessing import set_start_method
import yaml

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# kl divergence
def regularizer(mu, logvar):
    # it still returns a vector with dim: (batchsize,)
    return kl_weight * 2 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1)

criteriaMSE = torch.nn.MSELoss()


def criteria(recon_batch, init_batch):
    loss_xy = criteriaMSE(recon_batch[:, 0:2], init_batch[:, 0:2])
    yaw_diff = torch.abs(recon_batch[:, 2] - init_batch[:, 2]) # suppose it is within (0,2pi)
    yaw_diff_adjusted = torch.min(yaw_diff, np.pi*2-yaw_diff)
    loss_yaw = torch.mean(yaw_diff_adjusted**2)
    return loss_yaw + loss_xy


if __name__ == '__main__':
    
    with open('config.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # parameters 
    x_dim = cfg['cvae']['x_dim']
    y_dim_start = cfg['cvae']['y_dim_start']
    y_dim = cfg['cvae']['y_dim']
    dim_in = cfg['cvae']['dim_in']
    dim_latent = cfg['cvae']['dim_latent']
    dim_hidden = cfg['cvae']['dim_hidden']
    max_epochs = cfg['max_epochs']
    kl_weight = cfg['kl_weight']
    batch_size = cfg['batch_size']
    shuffle = cfg['shuffle']
    num_workers = cfg['num_workers']
    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}
    
    # Datasets
    data = np.load('data_random.npy') 
    
    data_training = data[0:int(0.8 * len(data))]
    data_validation = data[int(0.8 * len(data)):]
    data_training_set = Dataset(data_training)
    data_training_generator = torch.utils.data.DataLoader(data_training_set, **params)
    data_validation_set = Dataset(data_validation)
    data_validation_generator = torch.utils.data.DataLoader(data_validation_set, **params)

    encoder = Encoder(x_dim + y_dim, dim_hidden, dim_latent)
    decoder = Decoder(dim_latent + y_dim, dim_hidden, x_dim)
    model = CVAE(encoder, decoder)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # logging
    description = "_kl_" + str(kl_weight) + "_ite_" + str(max_epochs)
    folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + description
    writer = SummaryWriter("runs/" + folder)
    os.makedirs(name="policies", exist_ok=True)
    os.makedirs(name="policies/" + folder)

    try:
        print("========= start training ============")
        for epoch in range(max_epochs):
            epoch_loss, recon_loss_train = [], []
            for local_batch in data_training_generator:
                x_train = local_batch[:, 0:x_dim].type(dtype).to(device)
                y_train = local_batch[:, y_dim_start:].type(dtype).to(device)
                optimizer.zero_grad()
                mu, logvar, recon_batch = model(x_train, y_train)
                recon_loss = criteria(recon_batch, x_train)
                kl_loss = regularizer(mu, logvar)
                loss = torch.mean(recon_loss + kl_loss, 0)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()) # get loss.value
                recon_loss_train.append(recon_loss.item())
            writer.add_scalar('training/loss', np.mean(np.array(epoch_loss)), epoch)
            print("epoch: {}, Loss: {}, Recon_loss:{}, weight * KL_loss:{}".format(epoch,
                                                                                   np.mean((np.array(epoch_loss))),
                                                                                   np.mean(np.array(recon_loss_train)),
                                                                                   np.mean((np.array(epoch_loss))) -
                                                                                   np.mean(np.array(recon_loss_train))))
            # validation
            if epoch % 100 == 0 and epoch > 0:
                model.eval()
                recon_loss_viz, kl_loss_viz, loss_viz = [], [], []
                for local_batch in data_validation_generator:
                    x_validate = local_batch[:, 0:x_dim].type(dtype).to(device)
                    y_validate = local_batch[:, y_dim_start:].type(dtype).to(device)
                    mu, logvar, recon_batch = model(x_validate, y_validate)
                    recon_loss = criteria(recon_batch, x_validate)
                    kl_loss = regularizer(mu, logvar)
                    loss = torch.mean(recon_loss + kl_loss, 0)
                    recon_loss_viz.append(recon_loss.item())
                    loss_viz.append(loss.item())
                    kl_loss_viz.append(loss.item() - recon_loss.item())
                writer.add_scalar('validation/recon_loss', np.mean(np.array(recon_loss_viz)), epoch)
                writer.add_scalar('validation/weighted_kl_loss', np.mean((np.array(kl_loss_viz))), epoch)
                writer.add_scalar('validation/loss', np.mean((np.array(loss_viz))), epoch)
                print("epoch: {}, Loss: {}, Recon_loss: {}, weight * KL_loss: {}".format(epoch,
                                                                                np.mean((np.array(loss_viz))),
                                                                                np.mean(np.array(recon_loss_viz)),
                                                                                np.mean((np.array(kl_loss_viz)))))
                model.train()
                
            # save intermediate model
            if epoch % 2000 == 0 and epoch > 0:
                print("Saving intermediate model.")
                save_path = "policies/" + folder +"/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_intermediate.pt"
                torch.save(model, f=save_path)

        # save final model
        print("Finishing training, Saving final model.")
        save_path = "policies/" + folder + "/final.pt"
        torch.save(model, f=save_path)

    except KeyboardInterrupt:
        print("Training interrupted, saving final model.")
        save_path = "policies/" + folder + "/final.pt"
        torch.save(model, f=save_path)

    writer.close()
