import torch
from model_def import ModelNN, GainPredictor, MapEncoder, PoseEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

############ data packing ############

data = np.load('sample_22_02_2022.npy') # 200,000 pieces
y_data = data[:,0]
pose_data = data[:,2:6]
map_data = data[:,6:631].reshape(-1,1,25,25) # (batch, channel, dim1, dim2)

map_train, map_test, pose_train, pose_test, y_train, y_test = train_test_split(map_data, pose_data, y_data, test_size=0.9)

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

# test data    
class testData(Dataset):
    def __init__(self, map_data, pose_data):
        self.map_data = map_data
        self.pose_data = pose_data
    def __getitem__(self, index):
        return self.map_data[index], self.pose_data[index]
    def __len__ (self):
        return len(self.map_data)

train_data = trainData(torch.FloatTensor(map_train), torch.FloatTensor(pose_train), torch.FloatTensor(y_train))
test_data = testData(torch.FloatTensor(map_test[0:100]), torch.FloatTensor(pose_test[0:100]))
train_loader = DataLoader(train_data, batch_size=256)
test_loader = DataLoader(test_data)


############ model training ############

device = 'cpu'
EPOCHS = 1

model = ModelNN()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
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

############ model saving ############

torch.save(model.map_encoder, "map_encoder.pt")
torch.save(model.pose_encoder, "pose_encoder.pt")
torch.save(model.gain_predictor, "gain_predictor.pt")
torch.save(model, "model.pt")

############ model evaluating ############

y_test_pred = []
model.eval()
with torch.no_grad():
    for map_batch, pose_batch in test_loader:
        map_batch, pose_batch = map_batch.to(device), pose_batch.to(device)
        y_pred = float(model(map_batch, pose_batch).reshape(-1))
        y_test_pred.append(y_pred)
print(y_test_pred)

