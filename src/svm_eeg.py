'''

SUPPORT VECTOR MACHINES ON EEG DATASET

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

# %% LOAD DATASET

datafile = 'EEG'
dataset = utils.EEG_Data()

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels,
    split_fraction=0.5)

train_samples -= 1
test_samples -= 1

num_train_samples, input_dim = train_samples.shape
output_dim = len(np.unique(train_labels))

# %% MODEL PARAMETERS

num_classes = 5
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

os.makedirs('./../results/ex5', exist_ok=True)
path = './../results/ex5/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_SVM_dataset_' + datafile
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

# %%
