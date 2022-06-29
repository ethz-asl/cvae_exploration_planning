import numpy as np
import torch
import torch.optim as optim
from torch.multiprocessing import set_start_method
from tensorboardX import SummaryWriter
import datetime
import os
import yaml
import sys
from data import Dataset
from model import Encoder, Decoder, CVAE
from util import criteria, kl_regularizer

try:
    set_start_method('spawn')
except RuntimeError:
    pass

REPOSITORY_ROOT = os.path.abspath(
    os.path.join(__file__, os.path.pardir, os.pardir))


def main():
    print("Setting up CVAE training...")
    with open('config_cvae.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Training parameters.
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
    data_path = cfg['data_path']

    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}

    # Load the dataset.
    if not data_path.startswith('/'):
        data_path = os.path.join(REPOSITORY_ROOT, data_path)

    if not os.path.isfile(data_path):
        print(f"Dataset file '{data_path}' does not exist!")
        sys.exit(1)
    data = np.load(data_path)

    data_training = data[0:int(0.8 * len(data))]
    data_validation = data[int(0.8 * len(data)):]
    data_training_set = Dataset(data_training)
    data_training_generator = torch.utils.data.DataLoader(
        data_training_set, **params)
    data_validation_set = Dataset(data_validation)
    data_validation_generator = torch.utils.data.DataLoader(
        data_validation_set, **params)

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
                kl_loss = kl_regularizer(mu, logvar, kl_weight)
                loss = torch.mean(recon_loss + kl_loss, 0)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())  # get loss.value
                recon_loss_train.append(recon_loss.item())
            writer.add_scalar(
                'training/loss', np.mean(np.array(epoch_loss)), epoch)
            print("epoch: {}, Loss: {}, Recon_loss:{}, weight * KL_loss:{}".format(epoch, np.mean((np.array(epoch_loss))),
                  np.mean(np.array(recon_loss_train)), np.mean((np.array(epoch_loss))) - np.mean(np.array(recon_loss_train))))
            # validation
            if epoch % 100 == 0 and epoch > 0:
                model.eval()
                recon_loss_viz, kl_loss_viz, loss_viz = [], [], []
                for local_batch in data_validation_generator:
                    x_validate = local_batch[:, 0:x_dim].type(dtype).to(device)
                    y_validate = local_batch[:, y_dim_start:].type(
                        dtype).to(device)
                    mu, logvar, recon_batch = model(x_validate, y_validate)
                    recon_loss = criteria(recon_batch, x_validate)
                    kl_loss = kl_regularizer(mu, logvar, kl_weight)
                    loss = torch.mean(recon_loss + kl_loss, 0)
                    recon_loss_viz.append(recon_loss.item())
                    loss_viz.append(loss.item())
                    kl_loss_viz.append(loss.item() - recon_loss.item())
                writer.add_scalar('validation/recon_loss',
                                  np.mean(np.array(recon_loss_viz)), epoch)
                writer.add_scalar('validation/weighted_kl_loss',
                                  np.mean((np.array(kl_loss_viz))), epoch)
                writer.add_scalar('validation/loss',
                                  np.mean((np.array(loss_viz))), epoch)
                print("epoch: {}, Loss: {}, Recon_loss: {}, weight * KL_loss: {}".format(epoch, np.mean(
                    (np.array(loss_viz))), np.mean(np.array(recon_loss_viz)), np.mean((np.array(kl_loss_viz)))))
                model.train()

            # save intermediate model
            if epoch % 2000 == 0 and epoch > 0:
                print("Saving intermediate model.")
                save_path = "policies/" + folder + "/" + \
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_intermediate.pt"
                torch.save(model, f=save_path)

        # save final model
        print("Finishing training, Saving final model.")
        save_path = "policies/" + folder + "/final.pt"
        torch.save(model, f=save_path)

    except KeyboardInterrupt:
        print("Training interrupted, saving final model.")
        save_path = "policies/" + folder + "/final_interrupted.pt"
        torch.save(model, f=save_path)

    writer.close()


if __name__ == '__main__':
    main()
