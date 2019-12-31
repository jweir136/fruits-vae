import os

os.system("pip install tqdm")

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

from FruitVAE import FruitVAE # import the model

ROOT_DIR = "/storage/fruit/fruits-360/fruits-360_dataset/fruits-360"
TRAINING_DIR = os.path.join(ROOT_DIR, "Training")
TEST_DIR = os.path.join(ROOT_DIR, "Test")

############################ LOAD THE DATA ##################################################

"""
 [!] NOTE: The mean and std values were commuted in advance.
"""

mean = np.load("mean.npy")
std = np.load("std.npy")

trans = transforms.Compose([
  transforms.Resize(100),
  transforms.ToTensor(),
  transforms.Normalize(mean, std)
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, batch_size=32, shuffle=True, num_workers=12)

testfolder = datasets.ImageFolder(root=TEST_DIR, transform=trans)
testloader = data.DataLoader(testfolder, batch_size=32, shuffle=True, num_workers=12)

########################### LOAD THE MODEL, CREATE THE LOSS, AND INIT THE OPTIMIZER ################################################

def loss_function(x_pred, x, mu, logvar):
  mse = fn.mse_loss(x_pred, x)
  kl = -5e-4 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return mse + kl

vae = FruitVAE().cuda()
sgd = optim.SGD(vae.parameters(), lr=1e-3, momentum=0.9)

########################## TRAIN THE MODEL #########################################################################################

for epoch in range(20):
  for x, _ in tqdm(trainloader):
    x = x.cuda().float()
    sgd.zero_grad()

    x_pred, mu, logvar = vae.forward(x)
    train_loss = loss_function(x_pred, x, mu, logvar)

    train_loss.backward()
    sgd.step()

  for x, _ in tqdm(testloader):
    with torch.zero_grad():
      x = x.cuda().float()
      x_pred, mu, logvar = vae.forward(x)
      test_loss = loss_function(x_pred, x, mu, logvar)

  print("\n")
  print("[{}] Train Loss={} Test Loss={}".format(epoch+1, train_loss.detach().cpu().numpy(), test_loss.detach().cpu().numpy()))
  print("\n")

  if epoch+1 % 5 == 0:
    torch.save(vae.state_dict(), "/artifacts/epoch-{}-weights.pth".format(epoch+1))

  # Create a fake generation.
  with torch.no_grad():
    sample = torch.randn(1, 2).cuda().float()
    img = vae.decode(sample).detach().cpu().numpy()[0]
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    plt.show()

torch.save(vae.state_dict(), "/artifacts/model-weights.pth")
