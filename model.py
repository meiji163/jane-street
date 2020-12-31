import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from math import sqrt


def synth_data( fts, means, stds):
    '''Fill in missing features (0 -- 129) with samples from MLE Gaussian
    args:
        features: (n, 130) np.array
        means: (130,) means of features
        stds: (130,) standard deviations of features'''
    nan = np.nonzero(np.where(fts != fts, 1., 0.))
    for i, j in zip(*nan):
        fts[i,j] = np.random.normal(loc = means[j], scale = stds[j])
    return fts 
 
def TradingData(Dataset):
    '''Training dataset from numpy zipped file.
    kwargs:
        synth (bool): if True, synthesize missing data
        ts (bool): if True, process into time series data
        seq_len (int): sequence length of time series data'''
    def __init__(self, npz_path, **kwargs): 
        self.ts = kwargs.get("seq", True)
        self.synth = kwargs.get("synth", True)
        self.seq_len = kwargs.get("seq_len", 50)
        
        df = np.load(npz_path) 
        states = df["state"]
        means = df["mean"]
        stds = df["std"]
        self.time = torch.from_numpy(df["date"])
        if synth:
            self.data = torch.from_numpy(synth_data(states[:,:-1], means, stds))
        else:
            self.data = torch.from_numpy(states)
        self.meta = torch.from_numpy(df["meta"]).T
        self.weight = torch.from_numpy(states[:,-1])

    def __len__(self):
        if self.ts:
            return self.data.shape[0]
        return self.data.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        if not self.seq:
            return self.data[idx], self.weight[idx]
        return self.data[idx:idx+seq_len, :], self.weight[idx:idx+seq_len,:] 
         

def u(action, weight, n_dates, resp):
    '''Utility/reward function for trading actions
    args:
        action (n, 1) trade or not 0/1
        weight (n, 1) tensor
        n_dates (int): number of unique dates
        resp (n, 1): returns for trades
    '''
    P = torch.zeros(date_set.size())
    for d in date_set:
        idx = torch.nonzero(torch.where(date == d, 1.,0.))
        P[d] = (action[idx]*weight[idx]*resp[idx]).sum()
    t = math.sqrt(250./date.size(0))* P.sum()/torch.norm(P)
    return min(max(t.item(),0), 6)* P.sum()

class TradingVAE(nn.Module):
    def __init__(self, meta, latent_dim = 16):
        '''
        Variational autoencoder for Jane Street trading data
        args:
            latent_dim (int): number of latent variables
            meta: (29, 130) tensor of meta-features
        '''
        super(TradingVAE, self).__init__()
        self.dim = latent_dim
        self.relu = nn.ReLU()

        # meta-features contributes bias to corresponding features 
        self.meta = meta.unsqueeze(0)
        self.conv= Conv1dUntiedBias(130,29,1,1) 
        self.meta_bias = None
        #encoder
        self.lin1 = nn.Linear(130, 90) 
        self.lin_mu = nn.Linear(90, self.dim)
        self.lin_logvar = nn.Linear(90, self.dim)
        #decoder 
        self.lin2 = nn.Linear(self.dim, 90)
        self.lin3 = nn.Linear(90,130)

    def encode(self, x):
        '''
        args:
            x (n, 130) tensor: (batch, feature) 
        '''
        if self.training or self.meta_bias is None:
            self.meta_bias = self.relu(self.conv(self.meta)).view(1,-1)
        x += self.meta_bias.repeat(x.size(0), 1)
        y = self.relu(self.lin1(x))
        return self.lin_mu(y), self.lin_logvar(y)

    def decode(self, z):
        x = self.relu(self.lin2(z))
        x = self.lin3(x)
        return x   

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            e = torch.randn_like(std)
            return std*e + mu
        else:
            return mu

    def forward(self, x):
        x = x.to(torch.float32)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar 

def VAEloss(recons, inputs,
            mu, logvar, KL_weight = 1.0):
    recon_loss = F.mse_loss(recons, inputs)
    D_KL = -0.5*torch.sum(1 + logvar - mu**2 -logvar.exp())
    return (recon_loss + KL_weight*D_KL, recon_loss)

class TradingRNN(nn.Module):
    def __init__(self):
        self.embedding = ...
        self.rnn = nn.GRU(131,256, 
        self.lin = nn.Linear()


class Conv1dUntiedBias(nn.Module):
    def __init__(self, size, in_channels, out_channels,
                kernel_size, stride = 1, padding = 0,
                dilation = 1, groups = 1): 
        super(Conv1dUntiedBias, self).__init__()
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        kernel_size = _single(kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels, size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./sqrt(self.in_channels*self.kernel_size[0])
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x): 
        y = F.conv1d(x, self.weight, None, self.stride,
                    self.padding, self.dilation, self.groups)
        y += self.bias.unsqueeze(0).repeat(x.size(0),1,1)
        return y 

