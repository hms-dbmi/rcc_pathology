# cdkn2a_classifier_hypopt.py
# A binary classifier to distinguish patients with CDKN2A alterations (deletions or truncations)
# 
# 2020.12.21 Eliana Marostica 

# Parameters

subtype     = "KIRC"     # renal cell carcinoma subtype; one of 'KIRC' (clear cell), 'KIRP' (papillary), or 'KICH' (chromophobe)
model_type  = "Res50"    # one of "VGG16", "IncV3", or "Res50"
gpu_id      = "1"
date 			  = '20210122' # date or other distinguishing identifier
augment 		= True       # perform image augmentation

hdf5_path 		= './' + subtype + '_9pdeletion.hdf5'
experiment_name = subtype + '9pDeletion' + model_type + date


# GPU

import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#if gpu_id == 0 or gpu_id == 1: config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))



# Import packages

import h5py
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import genfromtxt
from numpy.random import seed

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from scipy.stats import sem
from collections import Counter
from tensorflow import set_random_seed
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from talos.model.normalizers import lr_normalizer
from talos.metrics.keras_metrics import f1score, precision, recall

from keras import losses
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3


# Hyperparameters

# for example:
param_dict = {'batch_size': [64], 
              'epochs': [15],
              'lr': [0.01],
              'class_weights': [True],
              'optimizer': [RMSprop]}



# Load data

hf = h5py.File(hdf5_path, 'r')
train_X = hf['train_img'][()]
train_y = hf['train_labels'][()].astype(int)
val_X = hf['val_img'][()]
val_y = hf['val_labels'][()].astype(int)
test_X = hf['test_img'][()]
test_y = hf['test_labels'][()].astype(int)

print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
print(test_X.shape)
print(test_y.shape)



# Define methods

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}


def deletion_model(x_train, y_train, x_val, y_val, params):
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
  # train the top added layers first, keep the VGG16 layers frozen
  # we can relax this restriction after the last layer is trained
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                loss=binary_crossentropy, 
                metrics=['accuracy', f1score, precision, recall])
  # train the model
  out = model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  callbacks=[EarlyStopping(monitor="val_f1score", patience=7, min_delta=0.001, mode='max')],
                  verbose=1,
                  class_weight=get_class_weights(y_train) if params['class_weights'] else None,
                  validation_data=[x_val, y_val])
  return out, model


def deletion_model_aug(x_train, y_train, x_val, y_val, params):
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
  # train the top added layers first, keep the VGG16 layers frozen
  # we can relax this restriction after the last layer is trained
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                loss=binary_crossentropy, 
                metrics=['accuracy', f1score, precision, recall])
  # Perform Data Augmentation while training
  batch_size = params['batch_size']
  out = model.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size, shuffle=False),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=params['epochs'],
                            validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size, shuffle=False),
                            validation_steps=x_val.shape[0] // batch_size,
                            class_weight=get_class_weights(y_train) if params['class_weights'] else None,
                            callbacks=[EarlyStopping(monitor="val_f1score", patience=7, min_delta=0.001, mode='max')])
  return out, model



# Image augmentation

train_generator = ImageDataGenerator(rescale=1.0/255.0,
	horizontal_flip=True, 
	vertical_flip=True, 
	rotation_range=90)
val_generator = ImageDataGenerator(rescale=1.0/255.0)


# Perform hyperparameters

scan_object = ta.Scan(x=train_X, y=train_y,
                      x_val=val_X, y_val=val_y,
                      val_split=0,
                      params=param_dict,
                      model=deletion_model_aug if augment else deletion_model,
                      experiment_name=experiment_name + '_scan',
                      print_params=True,
                      save_weights=True)



# Print or save hyperparameter optimization informtion

print("********Scan RESULTS*********")
print("Hypopt results:\n", scan_object.data)
print("Scan object details:\n", scan_object.details)
print("Time for each permutation:\n", scan_object.round_times)

analyze_object = ta.Analyze(scan_object)
print("********Analyze RESULTS*********")
print("Best hyperparameters:\n", analyze_object.best_params('val_acc', exclude=[], n=3, ascending=False))
analyze_object.correlate('val_loss', exclude=["epochs", "class_weights", "optimizer", "early_stopping", "losses"])
analyze_object.correlate('val_acc', exclude=["epochs", "class_weights", "optimizer", "early_stopping", "losses"])
# a regression plot for two dimensions 
plt.clf()
analyze_object.plot_regs('val_acc', 'val_loss')
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_val_loss_regs.png')
# a regression plot for two dimensions 
plt.clf()
analyze_object.plot_regs('loss', 'val_loss')
plt.savefig(experiment_name + '_scan/' + date + '_loss_val_loss_regs.png')
# line plot
plt.clf()
analyze_object.plot_line('val_acc')
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_line.png')
# kernel density estimator for val_accuracy
plt.clf()
analyze_object.plot_kde('val_acc')
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_kde.png')
# kernel density estimator for val_accuracy and batch_size
plt.clf()
analyze_object.plot_kde('batch_size', 'val_acc')
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_kde_batch_size.png')
# kernel density estimator for val_accuracy and learning rate
plt.clf()
analyze_object.plot_kde('lr', 'val_acc')
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_kde_lr.png')
# a simple histogram
plt.clf()
analyze_object.plot_hist('val_acc', bins=50)
plt.savefig(experiment_name + '_scan/' + date + '_val_acc_hist.png')
# heatmap correlation of validation accuracy
plt.clf()
analyze_object.plot_corr('val_acc', exclude=["epochs", "class_weights", "optimizer", "early_stopping"])
plt.savefig(experiment_name + '_scan/' + date + '_corr.png')
# a four dimensional bar grid
plt.clf()
analyze_object.plot_bars('lr', 'val_acc', 'batch_size', 'class_weights')
plt.savefig(experiment_name + '_scan/' + date + '_bar_grid.png')




# Make predictions on training and test data

predict_batch_size = 32
rescale_generator = ImageDataGenerator(rescale=1.0/255.0)
model = scan_object.best_model(metric='val_acc')

train_predict = model.predict_generator(generator=rescale_generator.flow(x=train_X, batch_size=predict_batch_size, shuffle=False), 
                                        steps=train_X.shape[0]/predict_batch_size, 
                                        verbose=1)
val_predict = model.predict_generator(generator=rescale_generator.flow(x=val_X, batch_size=predict_batch_size, shuffle=False), 
                                      steps=val_X.shape[0]/predict_batch_size,#ceildiv(val_X.shape[0], predict_batch_size),
                                      verbose=1)
test_predict = model.predict_generator(generator=rescale_generator.flow(x=test_X, batch_size=predict_batch_size, shuffle=False), 
                                       steps=test_X.shape[0]/predict_batch_size,
                                       verbose=1)




# Write train, val, and test predictions to file
for i, predictions in enumerate([train_predict, val_predict, test_predict]):
    split = ["train", "val", "test"][i]
    with open(experiment_name + '_' + split + 'Predictions.txt', 'w') as predictOutput:
        for i in range(len(predictions)):
            _ = predictOutput.write(f"{predictions[i][0]}\n")

for i, trues in enumerate([train_y, val_y, test_y]):
    split = ["train", "val", "test"][i]
    with open(experiment_name + '_' + split + 'True.txt', 'w') as trueOutput:
        for i in range(len(trues)):
            _ = trueOutput.write(f"{trues[i]}\n")


# Save best model

scan_object.x = np.zeros(500) # necessary for model to restore properly
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + "_hyperopt_models", metric="val_acc", asc=False)

