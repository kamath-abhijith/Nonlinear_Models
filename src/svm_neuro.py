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

# %% PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description = "SUPPORT VECTOR MACHINES FOR NEURO DATASET"
)

parser.add_argument('--parcellation', help="parcellation", default='tc_rest_aal')

args = parser.parse_args()

parcellation = args.parcellation

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

num_train_samples, _, dim1, dim2 = train_samples.shape
num_test_samples, _, dim1, dim2 = test_samples.shape

train_samples = train_samples.reshape(num_train_samples, dim1*dim2)
test_samples = dataset.samples.reshape(dataset.num_samples, dim1*dim2)
test_labels = dataset.labels

# %% MODEL PARAMETERS

num_classes = 2
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

os.makedirs('./../results/ex4', exist_ok=True)
path = './../results/ex4/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_SVM_dataset_' + parcellation
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=40,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)