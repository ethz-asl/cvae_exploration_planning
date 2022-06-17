import torch
import torch.nn.functional as F


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