'''

TOOLS FOR NONLINEAR MODELS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import numpy as np
import seaborn as sns
import torch

from scipy import io
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# %% PLOTTING

def plot_confusion_matrix(data, ax=None, xaxis_label=r'PREDICTED CLASS',
    yaxis_label=r'TRUE CLASS', map_min=0.0, map_max=1.0, title_text=None,
    show=True, save=False):
    ''' Plots confusion matrix '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    ax = sns.heatmap(data, vmin=map_min, vmax=map_max, linewidths=0.5,
        annot=True, fmt=".0f")
    # ax.invert_yaxis()
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_signal(x, y, ax=None, title_text=None, xaxis_label=None, 
    yaxis_label=None, plot_colour='blue', xlimits=[0,100], ylimits=[0,10],
    legend_label=None, legend_show=True, legend_loc='upper left',
    line_style='-', line_width=None, show=True, save=False):
    ''' Plots 1D samples for regression '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    plt.plot(x, y, color=plot_colour, linestyle=line_style,
        linewidth=line_width, label=legend_label)

    if legend_label and legend_show:
        plt.legend(ncol=1, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    # plt.ylim(ylimits)
    plt.xlim(xlimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% DATASETS

class Board(Dataset):
    
    def __init__(self, datafile):
        if datafile == 'Board0':
            data = np.loadtxt('./../data/Board/board_data.txt', delimiter=",",
                dtype=np.float32)
        elif datafile == 'Board10':
            data = np.loadtxt('./../data/Board/board_data_10.txt', delimiter=",",
                dtype=np.float32)
        elif datafile == 'Board25':
            data = np.loadtxt('./../data/Board/board_data_25.txt', delimiter=",",
                dtype=np.float32)

        self.samples = data[:,:2]
        self.labels = data[:, 2]
        self.num_samples = data.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class mnist(Dataset):

    def __init__(self, datafile, transform=None):
        
        if datafile == 'mnist':
            data = io.loadmat('./../data/MNIST/mnist_training_data.mat')
            labels = io.loadmat('./../data/MNIST/mnist_training_label.mat')

            self.num_samples = len(data['training_data'])
            self.samples = data['training_data'].reshape(self.num_samples, 28, 28)[:, None].astype(np.float32)
            self.labels = labels['training_label'][:,0].astype(np.int64)
            self.transform = transform

        elif datafile == 'mnist-rot':
            data = io.loadmat('./../data/MNIST/mnist-rot_training_data.mat')
            labels = io.loadmat('./../data/MNIST/mnist-rot_training_label.mat')

            self.num_samples = len(data['train_data'])
            self.samples = data['train_data'].reshape(self.num_samples, 28, 28)[:, None].astype(np.float32)
            self.labels = labels['train_label'][0,:].astype(np.int64)
            self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.samples[index]), self.labels[index]
        else:
            return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples


# %% DATA PROCESSING

def numpy_to_torch(data):
    return torch.from_numpy(data)

def train_test_splitter(samples, labels, split_fraction=0.5, state=12):
    return train_test_split(samples, labels, test_size=split_fraction,
        random_state=state)

def train_minibatches(data, batch_size=100, shuffle_flag=True):
    return DataLoader(dataset=data, batch_size=batch_size,
        shuffle=shuffle_flag)