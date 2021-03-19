# mutcount_regression_hypopt.py
# Performs regression of tumor mutation count values.
# 
# 2020.05.01 Eliana Marostica 

# Parameters

subtype         = "KIRC"
model_type      = "Res50"
gpu_id          = "1"
date 		= '20200609_2'

hdf5_path 	= '/mnt/data1/eliana/' + subtype + '_pancancer_mutcount.hdf5'
experiment_name = subtype + 'PanCancerMutCountRegression' + model_type


# GPU Setup

import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))


# Import required packages

import h5py
import math
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy.random import seed
from collections import Counter
from tensorflow import set_random_seed
from sklearn.preprocessing import normalize
from talos.model.normalizers import lr_normalizer
from sklearn.preprocessing import MinMaxScaler
from talos.metrics.keras_metrics import mse, mae, rmse
from keras.losses import binary_crossentropy

from keras import losses
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation

from scipy.stats import sem
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# Set parameters

param_dict = {'batch_size': [64], 
	      'epochs': [12],
	      'lr': [2,3,4],
	      'optimizer': [Adam],
              'losses': ['mean_squared_error'],
              'kernel_initializer': ['random_uniform'],
              'bias_initializer': ['glorot_normal']}


# Load data

hf = h5py.File(hdf5_path, 'r')
train_X = hf['train_img'][()]
train_y = hf['train_labels'][()]
val_X = hf['val_img'][()]
val_y = hf['val_labels'][()]
test_X = hf['test_img'][()]
test_y = hf['test_labels'][()]

print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
print(test_X.shape)
print(test_y.shape)


# Perform log transformation and normalization based on the training set

# reshape into 2d arrays for scaling
train_y = train_y.reshape(len(train_y), 1)
val_y = val_y.reshape(len(val_y), 1)
test_y = test_y.reshape(len(test_y), 1)

logmmscaler = MinMaxScaler()
logmmscaler.fit(np.log(train_y))
print(logmmscaler.data_max_)

# transform training dataset
train_y_lognorm = logmmscaler.transform(np.log(train_y))
#transform validation dataset
val_y_lognorm = logmmscaler.transform(np.log(val_y))
# transform test dataset
test_y_lognorm = logmmscaler.transform(np.log(test_y))


# Define methods

def save_labels(labels, file_root):
  '''Save continuous predictions or true values to text files.
  Args:
    labels   : list of training, val, and test label numpy arrays
    file_root: output filename root
  '''
  for i, labs in enumerate(labels):
    split = ["train", "val", "test"][i]
    with open(f"{file_root}_{split}.txt", 'w') as fileOutput:
        for i in range(len(labs)):
            _ = fileOutput.write(f"{labs.flatten()[i]}\n")


def spearman(y_true, y_pred):
  '''Calculate spearman correlation coefficient for use when training'''
  rho = ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32) )
  return tf.cond(tf.math.is_nan(rho), lambda: tf.cast(0, 'float32'), lambda: rho)


def mutcount_regression_aug_model(x_train, y_train, x_val, y_val, params):
  '''Input mutation count regression model with image augmentation to be submitted to talos.Scan
  
  Notes: 
    model_type is a global variable used to specify the base model for transfer learning
    train_generator and val_generator are also global variables
  '''
  if model_type == "VGG16":
    base_model = VGG16(weights='imagenet', include_top=False)  
  elif model_type == "Res50":
    base_model = ResNet50(weights='imagenet', include_top=False)
  else:
  	base_model = InceptionV3(weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='sigmoid', kernel_initializer=params['kernel_initializer'], 
            bias_initializer=params['bias_initializer'])(x)
  x = Dense(1)(x)
  model = Model(inputs=base_model.input, outputs=x)
  # train the top added layers first, keep the hidden layers frozen
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                loss=params['losses'], 
                metrics=[mae, rmse, spearman])
  # implement early stopping
  es = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.001, mode='min')
  # perform Data Augmentation while training
  batch_size = params['batch_size']
  out = model.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size, shuffle=False),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=params['epochs'],
                            validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size, shuffle=False),
                            validation_steps=x_val.shape[0] // batch_size,
                            callbacks=[es])
  return out, model



# Set random seed
import random
se = random.randint(0,100)
print("SEED: " + str(se))
random.seed(se)      # python
seed(se)             # numpy
set_random_seed(se)  # tensorflow


# Define generators for image augmentation and rescaling

train_generator = ImageDataGenerator(rescale=1.0/255.0,
  horizontal_flip=True, 
  vertical_flip=True, 
  rotation_range=90,
  zoom_range = 0.2,
  zca_whitening = True)
val_generator = ImageDataGenerator(rescale=1.0/255.0)



# Run Hyperparameter optimization

scan_object = ta.Scan(x=train_X, y=train_y_lognorm.flatten(),
                      x_val=val_X, y_val=val_y_lognorm.flatten(),
                      val_split=0,
                      params=param_dict,
                      model=mutcount_regression_aug_model,
                      experiment_name=experiment_name + '_scan',
                      print_params=True,
                      save_weights=True)


# Analyze hyperparameter optimization results

print("********Scan RESULTS*********")
print("Hypopt results:\n", scan_object.data)
print("Scan object details:\n", scan_object.details)
print("Time for each permutation:\n", scan_object.round_times)

analyze_object = ta.Analyze(scan_object)
print("********Analyze RESULTS*********")
print("Best hyperparameters:", analyze_object.best_params(metric="val_loss", n=1, exclude=[], ascending=True))

# a regression plot for two dimensions 
plt.clf()
analyze_object.plot_regs('val_loss', 'loss')
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_regs.png')
# line plot
plt.clf()
analyze_object.plot_line('val_loss')
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_val_mse_line.png')
# up to two dimensional kernel density estimator
plt.clf()
analyze_object.plot_kde('val_loss')
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_val_mse_kde.png')
# a simple histogram
plt.clf()
analyze_object.plot_hist('val_loss', bins=50)
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_val_mse_hist.png')
# heatmap correlation
plt.clf()
analyze_object.plot_corr('val_loss', ['val_loss', 'mse', 'loss'])
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_corr.png')
# a four dimensional bar grid
plt.clf()
analyze_object.plot_bars('lr', 'val_loss', 'kernel_initializer', 'bias_initializer')
plt.savefig(experiment_name + '_scan/' + date + 'lognorm_bar_grid.png')


# Gather and save predictions in the training, validation, and test set

predict_batch_size = 32
rescale_generator = ImageDataGenerator(rescale=1.0/255.0)
model = scan_object.best_model(metric='val_loss', asc=True)

train_predict_lognorm = model.predict_generator(generator=rescale_generator.flow(x=train_X, batch_size=predict_batch_size, shuffle=False), 
                                             steps=train_X.shape[0]/predict_batch_size, 
                                             verbose=1)
val_predict_lognorm = model.predict_generator(generator=rescale_generator.flow(x=val_X, batch_size=predict_batch_size, shuffle=False), 
                                           steps=val_X.shape[0]/predict_batch_size,
                                           verbose=1)
test_predict_lognorm = model.predict_generator(generator=rescale_generator.flow(x=test_X, batch_size=predict_batch_size, shuffle=False), 
                                            steps=test_X.shape[0]/predict_batch_size,
                                            verbose=1)

# predictions are in "log-normalized" space, so convert back to "tumor mutaton count" space
train_predict = logmmscaler.inverse_transform(train_predict_lognorm)
val_predict   = logmmscaler.inverse_transform(val_predict_lognorm)
test_predict  = logmmscaler.inverse_transform(test_predict_lognorm)

lognorm_predictions = [train_predict_lognorm, val_predict_lognorm, test_predict_lognorm]
lognorm_trues       = [train_y_lognorm, val_y_lognorm, test_y_lognorm]
predictions         = [train_predict, val_predict, test_predict]
trues               = [train_y, val_y, test_y]

save_labels(labels=lognorm_predictions, filename=f"{experiment_name}{date}PredictionsLogNorm")
save_labels(labels=lognorm_trues, filename=f"{experiment_name}{date}TruesLogNorm")
save_labels(labels=predictions, filename=f"{experiment_name}{date}Predictions")
save_labels(labels=trues, filename=f"{experiment_name}{date}Trues")


# Deploy best model

scan_object.x = np.zeros(500) # necessary for model to restore properly
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + date + "_hyperopt_models", metric="val_loss", asc=True)
