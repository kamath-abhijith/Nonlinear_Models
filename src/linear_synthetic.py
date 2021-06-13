'''

LINEAR CLASSIFICATION ON SYNTHETIC 2D DATA

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

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
    description = "LINEAR CLASSIFICATION ON SYNTHETIC DATA"
)

parser.add_argument('--dataset', help="training and testing dataset", default='noise0')
parser.add_argument('--training_size', help="training size", type=int, default=100)

args = parser.parse_args()

datafile = args.dataset
training_size = args.training_size

# %% LOAD DATASET

# datafile = 'noise0'
dataset = utils.Synthetic_2D(datafile)

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels)

num_train_samples, input_dim = train_samples.shape
num_classes = len(np.unique(train_labels))

# %% MODEL PARAMETERS

classifier = LogisticRegression()

# %% TRAINING

# training_size = 2000
np.random.seed(34)
random_idx = np.random.randint(num_train_samples, size=training_size)

classifier.fit(train_samples[random_idx], train_labels[random_idx])
print('TRAINING COMPLETE!')

# %% TESTING

# Predictions on test data
predictions = classifier.predict(test_samples)
confusion_mtx = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        confusion_mtx[i,j] = sum((test_labels==i) & (predictions==j))

accuracy = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100

# Predictions on mesh
num_samples = 500
x1, x2 = np.meshgrid(np.linspace(0, 2, num_samples), \
    np.linspace(0, 2, num_samples))
mesh_samples = np.array([x1, x2]).reshape(2, -1).T

mesh_labels = classifier.predict(mesh_samples).reshape(num_samples, num_samples)

# %% PLOTS

os.makedirs('./../results/ex1', exist_ok=True)
path = './../results/ex1/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'samples_LOG_dataset_' + datafile + '_size_' + str(training_size)
plt.contourf(x1, x2, mesh_labels, alpha=0.2, levels=np.linspace(0, 1, 3))
utils.plot_data2D(train_samples, train_labels, ax=ax, show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_LOG_dataset_' + datafile + '_size_' + str(training_size)
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=400,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

# %%
