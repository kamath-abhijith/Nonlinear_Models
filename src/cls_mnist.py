'''

CONVOLUTIONAL NEURAL NETWORK ON MNIST DATASET

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np

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

# %% PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description = "CONVOLUTIONAL NEURAL NETWORK FOR MNIST DATASET"
)

parser.add_argument('--dataset', help="training and testing dataset", default='mnist')
parser.add_argument('--method', help="method of optimisation", default='Adam')
parser.add_argument('--num_epochs', help="number of epochs", type=int, default=10)
parser.add_argument('--force_train', help="force training", type=bool, default=True)

args = parser.parse_args()

datafile = args.dataset
training_method = args.method
num_epochs = args.num_epochs
force_train = args.force_train

# %% LOAD DATASET

# Training set
dataset = utils.mnist(datafile=datafile)
num_train_samples = dataset.num_samples
output_dim = len(np.unique(dataset.labels))

if datafile == 'mnist':
    batch_size = 10000
elif datafile == 'mnist-rot':
    batch_size = 2000
train_loader = utils.train_minibatches(dataset, batch_size=batch_size)

# Testing set
if datafile == 'mnist':
    test_data = io.loadmat('./../data/MNIST/mnist_test_data.mat')
    test_labels = io.loadmat('./../data/MNIST/mnist_test_label.mat')

    num_test_samples = len(test_data['test_data'])
    test_samples = test_data['test_data'].reshape(num_test_samples, 28, 28)[:, None].astype(np.float32)
    test_labels = test_labels['test_label'][:,0].astype(np.int64)

elif datafile == 'mnist-rot':
    test_data = io.loadmat('./../data/MNIST/mnist-rot_test_data.mat')
    test_labels = io.loadmat('./../data/MNIST/mnist-rot_test_label.mat')

    num_test_samples = len(test_data['test_data'])
    test_samples = test_data['test_data'].reshape(num_test_samples, 28, 28)[:, None].astype(np.float32)
    test_labels = test_labels['test_label'][0,:].astype(np.int64)

test_samples = utils.numpy_to_torch(test_samples)
test_labels = utils.numpy_to_torch(test_labels)

# %% MODEL PARAMETERS

num_classes = 10
# training_method = 'Adam'
# num_epochs = 10
learning_rate = 0.01

model = nonlinear_tools.conv_network()
if training_method == 'SGD':
    learning_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif training_method == 'Adam':
    learning_rate = 0.01
    mom_weights = (0.9, 0.999)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
        betas=mom_weights)

# %% TRAINING

# force_train = False

os.makedirs('./../models/ex3', exist_ok=True)
path = './../models/ex3/'

if os.path.isfile(path + 'model_CNN_dataset_' + datafile + '_method_' + \
    str(training_method) + '.pth') and force_train==False:

    print('PICKING PRE-TRAINED MODEL')
    model = torch.load(path + 'model_CNN_dataset_' + datafile + '_method_' + \
        str(training_method) + '.pth')
    model.eval()

else:

    print('TRAINING IN PROGRESS...')

    training_error = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in (enumerate(train_loader)):

            # Forward pass
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_error[epoch] = loss.item()
        if (epoch+1) % 2 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, path + 'model_CNN_dataset_' + datafile + '_method_' + \
        str(training_method) + '.pth')
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

os.makedirs('./../results/ex3', exist_ok=True)
path = './../results/ex3/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_CNN_dataset_' + datafile + '_method_' + str(training_method)
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'loss_CNN_dataset_' + datafile + '_method_' + str(training_method)
utils.plot_signal(np.arange(1,num_epochs+1), training_error, ax=ax,
    xaxis_label=r'EPOCHS', yaxis_label=r'TRAINING ERROR', xlimits=[0,num_epochs],
    ylimits=[0,20], show=False, save=save_res)

# %%
