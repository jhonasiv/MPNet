import torch
from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2800, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(28, 128), nn.PReLU(), nn.Linear(128, 256), nn.PReLU(),
                                     nn.Linear(256, 512), nn.PReLU(), nn.Linear(512, 2800))
    
    def forward(self, x):
        x = self.decoder(x)
        return x


def loss_function(weights, x, recons_x, h):
    lambd = 1e-3
    mse = nn.MSELoss()
    mse_loss = mse(recons_x, x)
    """
    W is shape of N_hidden x N. So, we do not need to transpose it as opposed to
    http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    """
    dh = h * (1 - h)  # N_batch x N_hidden
    contractive_loss = torch.sum(dh ** 2 * torch.sum(Variable(weights) ** 2, dim=1), dim=1).mul_(lambd)
    return mse_loss + contractive_loss
