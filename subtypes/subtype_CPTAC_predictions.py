# subtype_CPTAC_predictions.py
# Get predictions of subtype classifier on CPTAC data
#
# 2019.09.12 Eliana Marostica


# Set parameters

subtype         = 'KIRC'      # CPTAC only contains clear cell data
model_type      = 'Res50'     # one of 'VGG16', 'IncV3', or 'Res50'
date            = ''          # the date and/or any additional information used to make a unique identifier
gpu_id          = '0'

hdf5_path 		= './' + subtype + '_bypid_cptac.hdf5'
talos_path     	= "./Subtype" + model_type + "_hyperopt_models.zip"
experiment_name = "./Subtype" + model_type + date



#################
# Setup
#################

# Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))


# Ensure version 2.1.2 is being used
import keras; print(keras.__version__)

#Import required packages
import sys
import h5py
import talos as ta
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

from scipy.stats import sem
from numpy import genfromtxt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix


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



#confusion matrix
y_true = val_y
all_predict = test_predict#[:,0]
y_pred = all_predict
y_pred_class = all_predict > 0.5
matrix = multilabel_confusion_matrix(y_true, y_pred_class)
print("KIRP Prediction Confusion Matrix")
print("tn: ", matrix[0].flatten()[0])
print("fp: ", matrix[0].flatten()[1])
print("fn: ", matrix[0].flatten()[2])
print("tp: ", matrix[0].flatten()[3])
print("KIRC Prediction Confusion Matrix")
print("tn: ", matrix[1].flatten()[0])
print("fp: ", matrix[1].flatten()[1])
print("fn: ", matrix[1].flatten()[2])
print("tp: ", matrix[1].flatten()[3])
print("KICH Prediction Confusion Matrix")
print("tn: ", matrix[2].flatten()[0])
print("fp: ", matrix[2].flatten()[1])
print("fn: ", matrix[2].flatten()[2])
print("tp: ", matrix[2].flatten()[3])


# Write all of the predictions and true values to output files

predictOutput = open(experiment_name + '_cptacPredictions.txt', 'w')
for row in range(len(all_predict)):
    predictOutput.write("%s\n" % str(all_predict.flatten()[row]))

predictOutput.close()

trueOutput = open(experiment_name + '_cptacTrue.txt', 'w')
for row in range(len(y_true)):
    trueOutput.write("%s\n" % str(y_true.flatten()[row]))

trueOutput.close()
