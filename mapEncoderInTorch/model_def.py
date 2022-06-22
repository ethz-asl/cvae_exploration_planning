import torch
import torch.nn as nn
import torch.nn.functional as F

############ NN definition ############

class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.map_encoder = MapEncoder()
        self.pose_encoder = PoseEncoder()
        self.gain_predictor = GainPredictor()
    
    def forward(self, x1, x2):
        xmap = self.map_encoder(x1)
        xpose = self.pose_encoder(x2)
        out = self.gain_predictor(xmap, xpose)
        return out
