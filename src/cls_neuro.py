'''

NEURAL NETWORK ON NEURO DATASET

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from scipy import io

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import nonlinear_tools
import utils

# %% PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description = "CONVOLUTIONAL NEURAL NETWORK FOR NEURO DATASET"
)

parser.add_argument('--parcellation', help="parcellation", default='tc_rest_aal')
parser.add_argument('--num_epochs', help="number of epochs", type=int, default=20)
parser.add_argument('--force_train', help="force training", type=bool, default=True)

args = parser.parse_args()

parcellation = args.parcellation
num_epochs = args.num_epochs
force_train = args.force_train

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% LOAD DATASET

# parcellation = 'tc_rest_power'
# parcellation = 'tc_rest_aal'
dataset = utils.Neuro_Dataset(parcellation)

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels,
    split_fraction=0.2)

train_samples = utils.numpy_to_torch(train_samples)
train_labels = utils.numpy_to_torch(train_labels).type(torch.LongTensor)

# test_samples = utils.numpy_to_torch(test_samples)
# test_labels = utils.numpy_to_torch(test_labels).type(torch.LongTensor)

test_samples = utils.numpy_to_torch(dataset.samples)
test_labels = utils.numpy_to_torch(dataset.labels).type(torch.LongTensor)

# %% MODEL PARAMETERS

num_classes = 2
# num_epochs = 100

if parcellation == 'tc_rest_aal':
    model = nonlinear_tools.neuroaal_network()
elif parcellation == 'tc_rest_power':
    model = nonlinear_tools.neuropower_network()
learning_rate = 0.01
optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

# %% TRAINING

# force_train = False

os.makedirs('./../models/ex4', exist_ok=True)
path = './../models/ex4/'

if os.path.isfile(path + 'model_CNN_dataset_' + parcellation + '.pth') and force_train==False:

    print('PICKING PRE-TRAINED MODEL')
    model = torch.load(path + 'model_CNN_dataset_' + parcellation + '.pth')
    model.eval()

else:
    print('TRAINING IN PROGRESS...')
    training_error = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):

        # Forward pass
        outputs = model(train_samples)
        loss = F.nll_loss(outputs, train_labels)

        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        training_error[epoch] = loss.item()
        if (epoch+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, path + 'model_CNN_dataset_' + parcellation + '.pth')
    print('TRAINING COMPLETE!')

# %% TESTING

with torch.no_grad():
    outputs = model(test_samples)
    _, predictions = torch.max(outputs.data, 1)
    confusion_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mtx[i,j] = sum((test_labels==i) & (predictions==j))

    accuracy = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100

# %% PLOTS

os.makedirs('./../results/ex4', exist_ok=True)
path = './../results/ex4/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_CNN_dataset_' + parcellation
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=40,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'loss_CNN_dataset_' + parcellation
utils.plot_signal(np.arange(1,num_epochs+1), training_error, ax=ax,
    xaxis_label=r'EPOCHS', yaxis_label=r'TRAINING ERROR', xlimits=[0,num_epochs],
    ylimits=[0,20], show=False, save=save_res)
# %%
