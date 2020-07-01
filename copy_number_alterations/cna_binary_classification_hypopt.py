# cna_binary_classification_hypopt.py
# Binary classification of CNA
# This file was created as a script for easy application to multiple genes in series or in parallel.
#
# 2020.03.02 Eliana Marostica
# 

import sys

if len(sys.argv) != 6:
    print('Incorrect number of arguments.\n',
          'Please provide subtype, model_type, gene, genenum, and gpu_id.')
    sys.exit(1)

else:
    subtype     = sys.argv[1]
    model_type  = sys.argv[2]
    gene        = sys.argv[3]
    genenum     = int(sys.argv[4])
    gpu_id      = sys.argv[5]

#ensure version 2.1.2 is being used
import keras; print(keras.__version__)

### Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.28 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))

#import required packages
import h5py
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy.random import seed
from collections import Counter
from tensorflow import set_random_seed
from sklearn.preprocessing import normalize
from talos.model.normalizers import lr_normalizer
from keras.losses import binary_crossentropy


from keras import losses
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation


from scipy.stats import sem
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# Set parameters

hdf5_path       = './' + subtype + '_cna2.hdf5'
experiment_name  = subtype + 'CNABinary' + gene + model_type

param_dict = {'batch_size': [64], #define parameters for talos hyperparameter tuning
              'epochs': [20],
              'early_stopping': [True],
              'class_weights': [True],
              'optimizer': ['Nadam']}


# Get numpy arrays of data

hf = h5py.File(hdf5_path, 'r')
train_X = hf['train_img'][()]
train_y = hf['train_labels'][:,genenum].astype(int)
val_X = hf['val_img'][()]
val_y = hf['val_labels'][:,genenum].astype(int)
test_X = hf['test_img'][()]
test_y = hf['test_labels'][:,genenum].astype(int)


# Print dimensions of data

print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
print(test_X.shape)
print(test_y.shape)


# Define methods

def get_class_weights(y):
  '''Get finary class weights, from https://github.com/keras-team/keras/issues/1875
  '''
  counter = Counter(y)
  majority = max(counter.values())
  return  {cls: float(majority/count) for cls, count in counter.items()}


def save_labels(labels, file_root):
  '''Save binary predictions or true values to text files.
  Args:
    labels   : list of training, val, and test label numpy arrays
    file_root: output filename root
  '''
  for i, labs in enumerate(labels):
    split = ["train", "val", "test"][i]
    with open(f"{file_root}_{split}.txt", 'w') as fileOutput:
        for i in range(len(labs)):
            _ = fileOutput.write(f"{labs.flatten()[i]}\n")


def cna_binary_model(x_train, y_train, x_val, y_val, params):
  '''Input CNA binary classification model to be submitted to talos.Scan
  Notes: model_type is a global variable used to specify the base model for transfer learning
  '''
  if model_type == "VGG16":
    base_model = VGG16(weights='imagenet', include_top=False)
  elif model_type == "Res50":
    base_model = ResNet50(weights='imagenet', include_top=False)
  else:
    base_model = InceptionV3(weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='sigmoid')(x)
  x = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=x)
  # train the top added layers first, keep the hidden layers frozen
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(optimizer=params['optimizer'], 
                loss=binary_crossentropy, 
                metrics=['accuracy'])
  batch_size = params["batch_size"]
  # train the model
  out = model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  class_weight=get_class_weights(y_train) if params['class_weights'] else None,
                  callbacks=[ta.utils.early_stopper(params['epochs'], mode="moderate")] if params['early_stopping'] else None,
                  verbose=1,
                  validation_data=[x_val, y_val])
  return out, model


# Run Hyperparameter optimization

scan_object = ta.Scan(x=train_X, y=train_y,
                      x_val=val_X, y_val=val_y,
                      val_split=0,
                      params=param_dict,
                      model=cna_binary_model,
                      experiment_name=experiment_name + '_scan',
                      print_params=True,
                      save_weights=True)


# Analyze hyperparameter optimization results

print("********Scan RESULTS*********")
print("Best model index:", scan_object.best_model(metric='val_acc'))
print("Hypopt results:\n", scan_object.data)
print("Scan object details:\n", scan_object.details)
print("Time for each permutation:\n", scan_object.round_times)
print("Round history:\n", scan_object.round_history)

a = ta.Analyze(scan_object)
print("********Analyze RESULTS*********")
print("Best hyperparameters:", a.best_params(metric="val_acc", n=1, exclude=[]))


# Save predictions in the training, validation, and test set

print("**********Predictions***********")
p = ta.Predict(scan_object, task="binary")
train_predict = p.predict(train_X, metric="val_acc")
val_predict   = p.predict(val_X, metric="val_acc")
test_predict  = p.predict(test_X, metric="val_acc")


predictions = [train_predict, val_predict, test_predict]
trues       = [train_y, val_y, test_y]

save_labels(labels=predictions, filename=f"{experiment_name}Predictions")
save_labels(labels=trues, filename=f"{experiment_name}Trues")


# Deploy best model

scan_object.x = np.zeros(500) # necessary for model to restore properly
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + "_hyperopt_models", metric="val_acc")
