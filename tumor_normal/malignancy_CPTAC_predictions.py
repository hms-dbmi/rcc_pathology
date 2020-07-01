# malignancy_CPTAC_predictions.py
# Get predictions of malignancy classifier on CPTAC data
#
# 2019.10.26 Eliana Marostica


# Set parameters

subtype         = 'KIRC'      # CPTAC only contains clear cell data
model_type      = 'Res50'     # one of 'VGG16', 'IncV3', or 'Res50'
date            = ''          # the date and/or any additional information used to make a unique identifier
gpu_id          = '0'

hdf5_path       = './' + subtype + '_bypid_cptac.hdf5'
talos_path      = "./KIRCTumorNormal" + model_type + "_hyperopt_models.zip"
experiment_name = "./KIRCTumorNormal" + model_type + date



#################
# Setup
#################

# Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))


# Ensure version 2.1.2 is being used
import keras; print(keras.__version__)

# Import required packages
import h5py
import talos as ta
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem
from numpy import genfromtxt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#################
# Get predictions
#################

# Get CPTAC images and labels

hf        = h5py.File(hdf5_path, 'r')
val_X     = hf['val_img'][()]
val_y     = hf['val_labels'][()]


# Print data dimensions

print(val_X.shape)
print(val_y.shape)


# Restore talos model

model = ta.Restore(talos_path)

test_predict = model.model.predict(val_X)


# Confusion matrix

print("CONFUSION MATRIX:")
y_true = val_y
all_predict = test_predict[:,0]
y_pred = test_predict[:,0]
y_pred_class = test_predict[:,0] > 0.5
matrix = confusion_matrix(y_true, y_pred_class)
print("tn: ", matrix.flatten()[0])
print("fp: ", matrix.flatten()[1])
print("fn: ", matrix.flatten()[2])
print("tp: ", matrix.flatten()[3])


# Write all of the predictions and true values to output files

predictOutput = open(experiment_name + '_cptacPredictions.txt', 'w')
for row in range(len(all_predict)):
    predictOutput.write("%s\n" % str(all_predict.flatten()[row]))

predictOutput.close()

trueOutput = open(experiment_name + '_cptacTrue.txt', 'w')
for row in range(len(y_true)):
    trueOutput.write("%s\n" % str(y_true.flatten()[row]))

trueOutput.close()
