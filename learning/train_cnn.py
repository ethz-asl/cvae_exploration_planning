import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

############ NN definition ############

class MapEncoder(nn.Module):
    def __init__(self):
        super(MapEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.conv3 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(108, 16)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class PoseEncoder(nn.Module):
    def __init__(self):
        super(PoseEncoder,self).__init__()
        self.layer_1 = nn.Linear(4, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 16) 
        
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.layer_out(x))
        return x

class GainPredictor(nn.Module):
    def __init__(self):
        super(GainPredictor,self).__init__()
        self.layer_1 = nn.Linear(32, 128) 
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, x1, x2):
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.layer_out(x))
        return x

class ModelNN(nn.Module):
    def __init__(self):
        super(ModelNN,self).__init__()
        self.map_encoder = MapEncoder()
        self.pose_encoder = PoseEncoder()
        self.gain_predictor = GainPredictor()
    
    def forward(self, x1, x2):
        xmap = self.map_encoder(x1)
        xpose = self.pose_encoder(x2)
        out = self.gain_predictor(xmap, xpose)
        return out

# train data structure
class trainData(Dataset):
    def __init__(self, map_data, pose_data, y_data):
        self.map_data = map_data
        self.pose_data = pose_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.map_data[index], self.pose_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.map_data)

# test data structure  
class testData(Dataset):
    def __init__(self, map_data, pose_data):
        self.map_data = map_data
        self.pose_data = pose_data
    def __getitem__(self, index):
        return self.map_data[index], self.pose_data[index]
    def __len__ (self):
        return len(self.map_data)

if __name__ == '__main__':

    ############ data packing ############

    data = np.load('sample_22_02_2022.npy') # 200,000 pieces
    y_data = data[:,0]
    pose_data = data[:,2:6]
    map_data = data[:,6:631].reshape(-1,1,25,25) # (batch, channel, dim1, dim2)

    map_train, map_test, pose_train, pose_test, y_train, y_test = train_test_split(map_data, pose_data, y_data, test_size=0.2)

    train_data = trainData(torch.FloatTensor(map_train), torch.FloatTensor(pose_train), torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(map_test), torch.FloatTensor(pose_test))
    train_loader = DataLoader(train_data, batch_size=16)
    test_loader = DataLoader(test_data)


    ############ model training ############

    device = 'cpu'
    EPOCHS = 200

    model = ModelNN()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    try:
        for e in range(1, EPOCHS+1):
            epoch_loss = 0
            for map_batch, pose_batch, y_batch in train_loader:
                map_batch, pose_batch, y_batch = map_batch.to(device), pose_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                y_pred = model(map_batch, pose_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
    except KeyboardInterrupt:
        ############ model saving ############
        torch.save(model.map_encoder, "map_encoder.pt")
        torch.save(model.pose_encoder, "pose_encoder.pt")
        torch.save(model.gain_predictor, "gain_predictor.pt")
        torch.save(model, "cnn_model.pt")

    torch.save(model, "cnn_model.pt")

    ############ model evaluating ############
    # model = torch.load("experiments/models/cnn_model.pt")
    # model.eval()
    # cond = map_test[0].reshape(1,1,25,25)
    # pose = pose_test[1].reshape(1,4)
    # y_pred = float(model(torch.FloatTensor(cond), torch.FloatTensor(pose)).reshape(-1))
    # print(y_pred)

    print("===== model evaluating =====")
    y_test_pred = []
    model.eval()
    with torch.no_grad():
        for map_batch, pose_batch in test_loader:
            map_batch, pose_batch = map_batch.to(device), pose_batch.to(device)
            y_pred = float(model(map_batch, pose_batch).reshape(-1))
            y_test_pred.append(y_pred)

    mse = ((np.array(y_test_pred) - y_test)**2).mean(axis=0)

    plt.figure(figsize=(12,6))

    rang = np.linspace(0,700,700)
    plt.scatter(y_test_pred, y_test, s=5, color="blue")
    plt.scatter(rang,rang,s=3,color='red')
    plt.text(0,700,"Mean square error of new voxels: "+str(int(mse)))
    plt.xlabel("predicted new voxels")
    plt.ylabel("true new voxels")
    plt.savefig("result.png", dpi=200)
    print("mse = ",mse)

