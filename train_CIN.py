import os
import cv2
import sys
import time
import ipdb
import math
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from FPAE import *
import torch.nn as nn
import torch.utils.data
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix


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


train_X = np.load('./data/train_X.npy')
train_Y = np.load('./data/train_Y.npy')
test_X = np.load('./data/test_X.npy')
test_Y = np.load('./data/test_Y.npy')

seed=212
train_data = CustomTensorDataset(
        tensors=(torch.from_numpy(train_X.astype('float')).float(), 
            torch.from_numpy(train_Y.astype('int')).int()), transform=vflip)

test_data = TensorDataset(
     torch.from_numpy(test_X.astype('float')).float(),
     torch.from_numpy(test_Y.astype('int')).int())

train_loader = DataLoader(train_data, batch_size=16, num_workers=0,
                          pin_memory=True, shuffle=True, 
                          worker_init_fn=np.random.seed(seed))
test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                         pin_memory=True, shuffle=True,
                         worker_init_fn=np.random.seed(seed))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_classification_model(model, epoch):
    torch.save(model.state_dict(), "./weights/classifier_{}.pth".format(epoch))

def train_classifier(num_epoch, train_loader, model1, model2, classifier, optimizer, nll_loss, scheduler):
  # Training
  epoch = 0
  best_acc = 0
  best_loss = float('inf')
  for epoch in range(epoch, num_epoch):
      correct = 0
      cnt = 0
      for data in train_loader:
          img, gt_label = data[0].permute(0, 3, 1, 2)[:, 0, :].unsqueeze(1), data[1]
          img = img.cuda().float()
          gt_label = gt_label.cuda().long()
          # =================== forward =====================
          output_1, org, z = model1(img)
          output_2, org, z = model2(img)
          pred = classifier(torch.cat((torch.abs(output_1-org), 
                                       torch.abs(output_2-org), 
                                       torch.abs(output_1 + output_2 - 2*org)), dim=1))         
          loss = nll_loss(pred, gt_label.squeeze())
          # get the index of the max log-probability
          pred = pred.detach().argmax(dim=1, keepdim=True).squeeze()
          correct += pred.eq(gt_label.detach().view_as(pred)).sum().item()

          # =================== backward ====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      if epoch % 1 == 0:
          print('epoch [{}/{}], loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)'.
                format(epoch+1, num_epoch, loss.item(), correct,
                        len(train_loader.dataset),
                        100. * correct / len(train_loader.dataset)))
          # Save best inference model
          if best_acc < correct:
              save_classification_model(classifier, epoch + 1)
              best_acc = correct
  return classifier


## === Loading the pretrained Healthy FPAE ===
model_1 = FPN_Gray()
model_1.to('cuda')
model_1.load_state_dict(torch.load("./weights/label_0_best.pth"))
model_1.eval()

## === Loading the pretrained non-COVID Pneumonia FPAE ===
model_2 = FPN_Gray()
model_2.to('cuda')
model_2.load_state_dict(torch.load("./weights/label_1_best.pth"))
model_2.eval()

num_epoch = 10
num_classes = 3
pretrained = True

## === Training the CIN model ===
classifier = models.resnet18(pretrained=True)
num_ftrs = classifier.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
classifier.fc = nn.Linear(num_ftrs, num_classes)
classifier = classifier.to('cuda')

print(f'Total number of combined parameters with FPAE and CIN: {count_parameters(classifier) + count_parameters(model_2) + count_parameters(model_1)}')

nll_loss = nn.CrossEntropyLoss(weight=torch.tensor([2., 2., 250.]).to('cuda'))
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5, weight_decay=1e-3)

if not pretrained:
    # Trainig classifier (Commented for now and used our trained model)
    classifier = train_classifier(num_epoch=num_epoch, train_loader=train_loader,
                                 model1=model_1, model2=model_2, 
                                 classifier=classifier,
                                 optimizer=optimizer, nll_loss=nll_loss, scheduler=scheduler)
else:
    ##  === Loading the pretrained CIN Model ===
    classifier.load_state_dict(torch.load("./weights/classifier_best.pth"))
    classifier.eval()


##  === Testing the Classification Model ===
correct = 0
gt = []
pr = []
for data in test_loader:
    torch.cuda.empty_cache()
    img, gt_label = data[0].permute(0, 3, 1, 2)[:, 0, :].unsqueeze(1), data[1]
    img = img.cuda().float()
    gt_label = gt_label.cuda().long()
    output_1, org, z = model_1(img)
    output_2, org, z = model_2(img)
    pred = classifier(torch.cat((torch.abs(output_1-org),
                                 torch.abs(output_2-org),
                                 torch.abs(output_1 + output_2 - 2*org)), dim=1))
    # get the index of the max log-probability
    pred = pred.argmax(dim=1, keepdim=True).squeeze()
    correct += pred.eq(gt_label.view_as(pred)).sum().item()
    gt.append(gt_label.item())
    pr.append(pred.item())


## === Evaluation metrics ###
mapping = ['Healthy', 'non-COVID Pneumonia', 'COVID-19']
recall = recall_score(gt, pr, average='weighted')
class_wise_recall = recall_score(gt, pr, average=None)
print(f'Sensitivity of each class:\n{mapping[0]} = {class_wise_recall[0]:.4f} | {mapping[1]} = {class_wise_recall[1]:.4f} | {mapping[2]} = {class_wise_recall[2]:.4f}\n')
precision = precision_score(gt, pr, average='weighted')
class_wise_precision = precision_score(gt, pr, average=None)
print(f'PPV of each class:\n{mapping[0]} = {class_wise_precision[0]:.4f} | {mapping[1]} = {class_wise_precision[1]:.4f} | {mapping[2]} = {class_wise_precision[2]:.4f}\n')

lr_precision, lr_recall, lr_auc = [], [], []
for mm in range(num_classes):
    pre, rec, _ = precision_recall_curve([(f==mm)*1 for f in gt], [(f==mm)*1 for f in pr])
    lr_precision.append(pre)
    lr_recall.append(rec)
    lr_auc.append(auc(rec, pre))

lr_f1 = f1_score(gt, pr, average='weighted')
print('## === Dataset Evaluation Results ===')
print(f'AUC: {np.mean(lr_auc):0.4f}\nAccuracy: {correct / len(test_loader.dataset):0.4f}\nPrecision: {precision:0.4f}\nRecall: {recall:0.4f} \nF1-score: {lr_f1:0.4f}')

print(f'Confusion Matrix:\n {confusion_matrix(gt, pr)}')
