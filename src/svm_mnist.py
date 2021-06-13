'''

SUPPORT VECTOR MACHINES ON MNIST DATASET

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np

from tqdm import tqdm
from scipy import io
from sklearn import svm

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
parser = argparse.ArgumentParser(
    description = "SUPPORT VECTOR MACHINES ON BOARD DATASET"
)

parser.add_argument('--dataset', help="training and testing dataset", default='mnist')

args = parser.parse_args()

datafile = args.dataset

# %% LOAD DATASET

datafile = 'mnist-rot'
if datafile == 'mnist':
    train_data = io.loadmat('./../data/MNIST/mnist_training_data.mat')
    train_labels = io.loadmat('./../data/MNIST/mnist_training_label.mat')
    test_data = io.loadmat('./../data/MNIST/mnist_test_data.mat')
    test_labels = io.loadmat('./../data/MNIST/mnist_test_label.mat')

    num_train_samples = len(train_data['training_data'])
    train_samples = train_data['training_data'].astype(np.float32)
    train_labels = train_labels['training_label'][:,0].astype(np.int64)

    num_test_samples = len(test_data['test_data'])
    test_samples = test_data['test_data'].astype(np.float32)
    test_labels = test_labels['test_label'][:,0].astype(np.int64)

elif datafile == 'mnist-rot':
    train_data = io.loadmat('./../data/MNIST/mnist-rot_training_data.mat')
    train_labels = io.loadmat('./../data/MNIST/mnist-rot_training_label.mat')
    test_data = io.loadmat('./../data/MNIST/mnist-rot_test_data.mat')
    test_labels = io.loadmat('./../data/MNIST/mnist-rot_test_label.mat')

    num_train_samples = len(train_data['train_data'])
    train_samples = train_data['train_data'].reshape(num_train_samples,784).astype(np.float32)
    train_labels = train_labels['train_label'][0,:].astype(np.int64)

    num_test_samples = len(test_data['test_data'])
    test_samples = test_data['test_data'].reshape(num_test_samples,784).astype(np.float32)
    test_labels = test_labels['test_label'][0,:].astype(np.int64)

# %% MODEL PARAMETERS

num_classes = 10
param_C = 1.0

classifier = svm.SVC(C=param_C, kernel='rbf')

# %% TRAINING

classifier.fit(train_samples, train_labels)
print('TRAINING COMPLETE!')

# %% TESTING

predictions = classifier.predict(test_samples)
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
save_res = path + 'accuracy_SVM_dataset_' + datafile
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

# %%
