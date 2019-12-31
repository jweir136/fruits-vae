import torch
import torch.nn as nn
import torch.nn.functional as fn

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], 64*84*84)

class UnFlatten(nn.Module):
  def forward(self, x):
    return x.view(x.size()[0], 64, 84, 84)

class FruitVAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.drop = nn.Dropout2d(p=0.2, inplace=True)

    self.encoder = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(8, 16, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.ReLU(True),
      Flatten(),
      nn.Linear(64*84*84, 400),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(400, 2)
    self.logvar_layer = nn.Linear(400, 2)
    self.decoder = nn.Sequential(
      nn.Linear(2, 400),
      nn.ReLU(True),
      nn.Linear(400, 64*84*84),
      nn.ReLU(True),
      UnFlatten(),
      nn.ConvTranspose2d(64, 32, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 3, kernel_size=5),
      nn.Tanh()
    )

  def reparam_(self, mu, logvar):
    std = torch.exp(logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x, training=False):
    if training:
      x = self.drop(x)
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return mu, logvar

  def decode(self, x):
    return self.decoder(x)

  def forward(self, x, training=False):
    if training:
      x = self.drop(x)
    mu, logvar = self.encode(x)
    z = self.reparam_(mu, logvar)
    return self.decode(z), mu, logvar
