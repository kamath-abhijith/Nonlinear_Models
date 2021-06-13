'''

CONVOLUTIONAL NEURAL NETWORK ON MNIST DATASET

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

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% ARGUMENT PARSER

# %% LOAD DATASET

datafile = 'EEG'
dataset = utils.EEG_Data()

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels)

num_train_samples, input_dim = train_samples.shape
output_dim = len(np.unique(train_labels))

train_samples = utils.numpy_to_torch(train_samples)
test_samples = utils.numpy_to_torch(test_samples)
train_labels = utils.numpy_to_torch(train_labels).type(torch.LongTensor) - 1
test_labels = utils.numpy_to_torch(test_labels).type(torch.LongTensor) - 1

# %% MODEL PARAMETERS

hidden1_dim = 5000
hidden2_dim = 800
num_classes = output_dim
num_epochs = 100
training_method = 'Adam'

model = nonlinear_tools.deep_network(input_dim, hidden1_dim, hidden2_dim, output_dim)
criterion = nn.CrossEntropyLoss()

if training_method == 'SGD':
    learning_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif training_method == 'Adam':
    learning_rate = 0.1
    mom_weights = (0.9, 0.999)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
        betas=mom_weights)

# %% TRAINING

force_train = True
training_size = num_train_samples

os.makedirs('./../models/ex5', exist_ok=True)
path = './../models/ex5/'

if os.path.isfile(path + 'model_DN_dataset_' + datafile + '_size_' + \
    str(training_size) + '_method_' + str(training_method) + '.pth') and force_train==False:

    print('PICKING PRE-TRAINED MODEL')
    model = torch.load(path + 'model_DN_dataset_' + datafile + '_size_' + \
        str(training_size) + '_method_' + str(training_method) + '.pth')
    model.eval()

else:
    np.random.seed(34)
    random_idx = np.random.randint(num_train_samples, size=training_size)

    print('TRAINING IN PROGRESS...')

    training_error = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):

        # Forward pass
        outputs = model(train_samples[random_idx])
        loss = criterion(outputs, train_labels[random_idx])
        training_error[epoch] = loss.item()
        
        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if (epoch+1) % num_epochs/10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model, path + 'model_DN_dataset_' + datafile + '_size_' + \
        str(training_size) + '_method_' + str(training_method) + '.pth')
    print('TRAINING COMPLETE!')

# %% TESTING

with torch.no_grad():
    # Predictions on test data
    outputs = model(test_samples)
    _, predictions = torch.max(outputs.data, 1)
    confusion_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mtx[i,j] = sum((test_labels==i) & (predictions==j))

    accuracy = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100

# %% PLOTS

os.makedirs('./../results/ex5', exist_ok=True)
path = './../results/ex5/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_DN_dataset_' + datafile + '_size_' + str(training_size)\
    + '_method_' + str(training_method)
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'loss_DN_dataset_' + datafile + '_size_' + str(training_size)\
     + '_method_' + str(training_method)
utils.plot_signal(np.arange(1,num_epochs+1), training_error, ax=ax,
    xaxis_label=r'EPOCHS', yaxis_label=r'TRAINING ERROR', xlimits=[0,num_epochs],
    ylimits=[0,20], show=False, save=save_res)
# %%
