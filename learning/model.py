import torch
import torch.nn.functional as F

############ CVAE definition ############

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear_mu = torch.nn.Linear(H, D_out)
        self.linear_logvar = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.H = H

    # Encoder Q netowrk, approximate the latent feature with gaussian distribution,output mu and logvar
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        return self.linear_mu(x), self.linear_logvar(x)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    # Decoder P network, sampling from normal distribution and build the reconstruction
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        return self.linear4(x)


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _reparameterize(mu, logvar):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        eps = torch.randn_like(mu)
        return mu + torch.exp(logvar / 2) * eps

    def forward(self, state):
        mu, logvar = self.encoder(state)
        z = self._reparameterize(mu, logvar)
        return mu, logvar, self.decoder(z)


class CVAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(logvar / 2) * eps

    def forward(self, state, cond):
        x_in = torch.cat((state, cond), 1)
        mu, logvar = self.encoder(x_in)
        z = self._reparameterize(mu, logvar)
        z_in = torch.cat((z, cond), 1)
        return mu, logvar, self.decoder(z_in)


############ Baseline Gain Estimator and Regression definition ############

class GainEstimator(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(GainEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(dim_in, dim_hidden)
        self.linear2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = torch.nn.Linear(dim_hidden, dim_out)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, cond):
        x = torch.cat((x, cond), 1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        return self.linear3(x)

# regression on best samples, as function of conditioned local map
class Regression(torch.nn.Module):
    def __init__(self, dim_in, hidden, dim_out):
        super(Regression, self).__init__()
        self.linear1 = torch.nn.Linear(dim_in, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.linear3 = torch.nn.Linear(hidden, hidden)
        self.linear4 = torch.nn.Linear(hidden, dim_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, cond):
        x = F.relu(self.linear1(cond))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        return self.linear4(x)

############ CNN definition ############

class MapEncoder(torch.nn.Module):
    def __init__(self):
        super(MapEncoder,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 5)
        self.conv2 = torch.nn.Conv2d(3, 3, 5)
        self.conv3 = torch.nn.Conv2d(3, 3, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(108, 16)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class PoseEncoder(torch.nn.Module):
    def __init__(self):
        super(PoseEncoder,self).__init__()
        self.layer_1 = torch.nn.Linear(4, 64) 
        self.layer_2 = torch.nn.Linear(64, 64)
        self.layer_out = torch.nn.Linear(64, 16) 
        
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.layer_out(x))
        return x

class CnnGainEstimator(torch.nn.Module):
    def __init__(self):
        super(CnnGainEstimator,self).__init__()
        self.layer_1 = torch.nn.Linear(32, 128) 
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_out = torch.nn.Linear(64, 1) 
        
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x1, x2):
        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.layer_out(x))
        return x

class CNNGainEstimatorModel(torch.nn.Module):
    def __init__(self):
        super(CNNGainEstimatorModel,self).__init__()
        self.map_encoder = MapEncoder()
        self.pose_encoder = PoseEncoder()
        self.gain_predictor = CnnGainEstimator()
    
    def forward(self, x1, x2):
        xmap = self.map_encoder(x1)
        xpose = self.pose_encoder(x2)
        out = self.gain_predictor(xmap, xpose)
        return out