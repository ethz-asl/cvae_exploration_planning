import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import CNNGainEstimatorModel
import yaml
import os
import sys
from datetime import datetime


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(__file__, os.path.pardir, os.pardir))


# train data structure
class trainData(Dataset):
    def __init__(self, map_data, pose_data, y_data):
        self.map_data = map_data
        self.pose_data = pose_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.map_data[index], self.pose_data[index], self.y_data[index]

    def __len__(self):
        return len(self.map_data)


# test data structure
class testData(Dataset):
    def __init__(self, map_data, pose_data):
        self.map_data = map_data
        self.pose_data = pose_data

    def __getitem__(self, index):
        return self.map_data[index], self.pose_data[index]

    def __len__(self):
        return len(self.map_data)


def main():
    # ############ Read config ############
    print("Setting up CNN training...")
    with open('config_cnn.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = cfg['data_path']

    folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_cnn"
    os.makedirs(name="runs", exist_ok=True)
    os.makedirs(name="runs/" + folder)

    ############ Load the dataset ############
    if not data_path.startswith('/'):
        data_path = os.path.join(REPOSITORY_ROOT, data_path)

    if not os.path.isfile(data_path):
        print(f"Dataset file '{data_path}' does not exist!")
        sys.exit(1)
    data = np.load(data_path)

    y_data = data[:, 0]
    pose_data = data[:, 2:6]
    # (batch, channel, dim1, dim2)
    map_data = data[:, 6:631].reshape(-1, 1, 25, 25)

    map_train, map_test, pose_train, pose_test, y_train, y_test = train_test_split(
        map_data, pose_data, y_data, test_size=0.2)

    train_data = trainData(torch.FloatTensor(map_train), torch.FloatTensor(
        pose_train), torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(map_test),
                         torch.FloatTensor(pose_test))
    train_loader = DataLoader(train_data, batch_size=16)
    test_loader = DataLoader(test_data)

    ############ model training ############

    model = CNNGainEstimatorModel()
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    print("========== Started Training CNN Model ==========")
    try:
        for e in range(1, cfg['max_epochs']+1):
            epoch_loss = 0
            for map_batch, pose_batch, y_batch in train_loader:
                map_batch, pose_batch, y_batch = map_batch.to(
                    device), pose_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(map_batch, pose_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

    except KeyboardInterrupt:
        ############ model saving ############
        torch.save(model.map_encoder, f"runs/{folder}/map_encoder.pt")
        torch.save(model.pose_encoder, f"runs/{folder}/pose_encoder.pt")
        torch.save(model.gain_predictor, f"runs/{folder}/gain_predictor.pt")
        torch.save(model, f"runs/{folder}/cnn_model.pt")

    torch.save(model, f"runs/{folder}/cnn_model_final.pt")

    ############ model evaluating ############

    print("========== Evaluating CNN Model ==========")
    y_test_pred = []
    model.eval()
    with torch.no_grad():
        for map_batch, pose_batch in test_loader:
            map_batch, pose_batch = map_batch.to(device), pose_batch.to(device)
            y_pred = float(model(map_batch, pose_batch).reshape(-1))
            y_test_pred.append(y_pred)

    mse = ((np.array(y_test_pred) - y_test)**2).mean(axis=0)

    plt.figure(figsize=(12, 6))
    rang = np.linspace(0, 700)
    plt.scatter(y_test_pred, y_test, s=5, color="blue")
    plt.plot(rang, rang,'r-')
    plt.text(0, 700, "Mean square error of new voxels: "+str(int(mse)))
    plt.xlabel("predicted new voxels")
    plt.ylabel("true new voxels")
    plt.savefig(f"runs/{folder}/evaluation_result.png", dpi=200)
    print("mse = ", mse)
    print(f"CNN Model and evaluation is stored in 'runs/{folder}'")
    print("========== Finished ==========")

if __name__ == '__main__':
    main()
