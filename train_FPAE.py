import os
import cv2
import sys
import glob
import time
import ipdb
import torch
import numpy as np
from FPAE import *
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

## =============================
# CUSTOM DATALOADER
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        y = self.tensors[1][index]
        
        if self.transform and y.item() != 2:
            val = torch.randint(0, 3, (1,)).item()
            if val == 0:
                x = vflip(x)
            elif val == 1:
                x = hflip(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# Dataset with flipping tranformations
def vflip(tensor):
    """Flips tensor vertically.
    """
    tensor = tensor.flip(1)
    return tensor


def hflip(tensor):
    """Flips tensor horizontally.
    """
    tensor = tensor.flip(2)
    return tensor


def save_model(model, epoch, label):
    torch.save(model.state_dict(), "./weights/label_{}_FPAE_{}.pth".format(label, epoch))


def train(num_epoch, train_loader, model, optimizer, mse_loss, label=0):
    # FPAE training function for a given DataLoader
    epoch = 0
    best_loss = float('inf')
    for epoch in range(epoch, num_epoch):
      batch_img = torch.FloatTensor().to('cuda').float()
      for data in train_loader:
          img, gt_label = data[0].permute(0, 3, 1, 2)[:, 0, :].unsqueeze(1), data[1]
          img = nn.functional.interpolate(img.to('cuda').float(), 512)
          gt_label = gt_label.to('cuda').long()
          if gt_label.item() == label:
              batch_img = torch.cat((batch_img, img), dim=0)
         
          if batch_img.shape[0] == 32:
              # ===================forward=====================
              optimizer.zero_grad()
              output, org, z = model(img)
              _, _, z1 = model(output)
              loss = mse_loss(output, org) + mse_loss(z, z1)

              # ===================backward====================
              loss.backward()
              optimizer.step()
              batch_img = torch.FloatTensor().to('cuda').float()

      # ===================log========================
      if epoch % 1 == 0:
          print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epoch, loss.item()/32))
          # Save best inference model
          if best_loss > loss.item():
              save_model(model, epoch + 1, label)
              best_loss = loss.item()
              ind = epoch
    return model


def test(test_loader, model, label=0):
    # FPAE testing function for a given DataLoader
    for data in test_loader:
      img, gt_label = data[0].permute(0, 3, 1, 2)[:, 0, :].unsqueeze(1), data[1]
      img = nn.functional.interpolate(img.to('cuda').float(), 512)
      gt_label = gt_label.cuda().long()

      if gt_label.item() == label:
          # ===================forward=====================
          temp_img = img.clone()
          output, _, _ = model(img)
          temp_output = output.clone()
          plt.gray()
          plt.subplot(131)
          plt.imshow(img.squeeze().cpu().numpy())
          plt.axis('off')
          plt.title('Original')
          plt.subplot(132)
          plt.imshow(output.detach().squeeze().cpu().numpy())
          plt.axis('off')          
          plt.title('Reconstructed')
          plt.subplot(133)
          plt.imshow((img - output.detach()).squeeze().cpu().numpy())
          plt.axis('off')          
          plt.title('Difference')          
          plt.savefig(f'./label_{label}_FPAE.png')
          break
          

# Loading the numpy training and testing data
train_X = np.load('./data/train_X.npy')
train_Y = np.load('./data/train_Y.npy')
test_X = np.load('./data/test_X.npy')
test_Y = np.load('./data/test_Y.npy')

seed=212
train_data = CustomTensorDataset(
        tensors=(torch.from_numpy(train_X.astype('float')).float(), 
            torch.from_numpy(train_Y.astype('int')).int()), transform=vflip)

train_loader = DataLoader(train_data, batch_size=1,
                                           shuffle=True,
                                           worker_init_fn=np.random.seed(seed))

test_data = TensorDataset(
     torch.from_numpy(test_X.astype('float')).float(),
     torch.from_numpy(test_Y.astype('int')).int())

test_loader = DataLoader(test_data, batch_size=1,
                                          shuffle=True,
                                          worker_init_fn=np.random.seed(seed))

MAX_EPOCH = 50
pretrained = True

# Training and testing the model on Healthy data
model = FPN_Gray()
model.to('cuda')
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
if not pretrained:
    model = train(num_epoch=MAX_EPOCH, train_loader=train_loader, model=model, optimizer=optimizer, mse_loss=mse_loss, label=0)
    print('## ===== Training finished for Healthy FPAE ===== ##')
else:
    model.load_state_dict(torch.load("./weights/label_0_best.pth"))
    test(test_loader=test_loader, model=model, label=0)

# Training and testing the model on non-COVID Pneumonia data
model = FPN_Gray()
model.to('cuda')
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
if not pretrained:
    model = train(num_epoch=MAX_EPOCH, train_loader=train_loader, model=model, optimizer=optimizer, mse_loss=mse_loss, label=1)
    print('## ===== Training finished for non-COVID Pneumonia FPAE  ===== ##')
else:
    model.load_state_dict(torch.load("./weights/label_1_best.pth"))
    test(test_loader=test_loader, model=model, label=1)
