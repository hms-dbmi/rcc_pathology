# malignancy_binary_classification_hypopt.py
# Run hyperparameter optimization for TumorNormal (i.e. Malignancy) task
# 
# 2019.11.03 Eliana Marostica

# Set parameters

subtype         = 'KIRC'      # one of 'KIRC' (clear cell), 'KIRP' (papillary), or 'KICH' (chromophobe)
model_type      = 'IncV3'     # one of 'VGG16', 'IncV3', or 'Res50'
date            = '20200628'  # today's date and/or any additional information to make a unique identifier
gpu_id          = '1'


hdf5_path       = './' + subtype + '_bypid.hdf5'        # automatically generated for this task
experiment_name = subtype + "TumorNormal" + model_type  # automatically generated for this task


# Ensure version 2.1.2 is being used
import keras; print(keras.__version__)

# Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))

# Import required packages
import h5py
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.preprocessing import normalize
from talos.model.normalizers import lr_normalizer

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


# Define parameters for talos hyperparameter tuning

if model_type == "Res50":
  lrs = [0.1, 0.9, 0.99, 0.999]
else:
  lrs = [0.01, 0.001, 0.0001, 0.00001]
  

p = {'batch_size': [32], 
     'epochs': [10],
     'lr': lrs,
     'optimizer': [RMSprop, Adam],
     'losses': ['binary_crossentropy']}


# Get numpy arrays of data

hf = h5py.File(hdf5_path, 'r')
train_X = hf['train_img'][()]
train_y = hf['train_labels'][()]
val_X = hf['val_img'][()]
val_y = hf['val_labels'][()]
test_X = hf['test_img'][()]
test_y = hf['test_labels'][()]


# Print dimensions of data

print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
print(test_X.shape)
print(test_y.shape)


# Define methods

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


def malignancy_model(x_train, y_train, x_val, y_val, params):
  '''Input malignancy model to be submitted to talos.Scan
  Notes: model_type is a global variable used to specify the base model for transfer learning
  '''
  if model_type == "VGG16":
    base_model = VGG16(weights='imagenet', include_top=False)
  elif model_type == "Res50":
    base_model = ResNet50(weights='imagenet', include_top=False)
  elif model_type == "IncV3":
    base_model = InceptionV3(weights='imagenet', include_top=False)
  else:
    raise Exception('Did not select a viable base model. Try VGG16, IncV3, or Res50.')
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='sigmoid')(x)
  x = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=x)
  # train the top added layers first, keep the hidden layers frozen
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                loss=params['losses'], 
                metrics=['accuracy'])
  batch_size = params["batch_size"]
  # train the model
  out = model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=1,
                  validation_data=[x_val, y_val])
  return out, model



# Run Hyperparameter optimization

scan_object = ta.Scan(x=train_X, y=train_y, 
                      x_val=val_X, y_val=val_y,
                      val_split=0,
                      params=p,
                      model=malignancy_model,
                      experiment_name = experiment_name + '_scan/')



# Analyze hyperparameter optimization results

print("********Scan RESULTS*********")
print("Hypopt results:\n", scan_object.data)
print("Scan object details:\n", scan_object.details)
print("Time for each permutation:\n", scan_object.round_times)

analyze_object = ta.Analyze(scan_object)
print("********Analyze RESULTS*********")
print("Best hyperparameters:", analyze_object.best_params(metric="val_acc", n=1, exclude=[], ascending=False))

# a regression plot for two dimensions 
plt.clf()
analyze_object.plot_regs('val_acc', 'acc')
plt.savefig(experiment_name + '_scan/' + date + '_regs.png')

# line plot
plt.clf()
analyze_object.plot_line('val_acc')
plt.savefig(experiment_name + '_scan/' + date + 'val_acc_line.png')

# up to two dimensional kernel density estimator
plt.clf()
analyze_object.plot_kde('val_acc')
plt.savefig(experiment_name + '_scan/' + date + 'val_acc_kde.png')

# a simple histogram
plt.clf()
analyze_object.plot_hist('val_acc', bins=50)
plt.savefig(experiment_name + '_scan/' + date + 'val_acc_hist.png')



# Save predictions in the training, validation, and test set

p = ta.Predict(scan_object, task="binary")
train_predict = p.predict(train_X, metric="val_acc", asc=False)
val_predict   = p.predict(val_X, metric="val_acc", asc=False)
test_predict  = p.predict(test_X, metric="val_acc", asc=False)

predictions = [train_predict, val_predict, test_predict]
trues       = [train_y, val_y, test_y]

save_labels(labels=predictions, filename=f"{experiment_name}{date}Predictions")
save_labels(labels=trues, filename=f"{experiment_name}{date}Trues")



# Deploy best model

scan_object.x = np.zeros(500)
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + date + '_hyperopt_models', metric="val_acc", asc=False)

