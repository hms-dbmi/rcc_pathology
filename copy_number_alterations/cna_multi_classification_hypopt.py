# cna_classifier_hypopt.py
# A multilabel classifier to predict CNA 
# 
# 2020.05.06 Eliana Marostica 

subtype         = "KIRC"
model_type      = "Res50"     # one of 'VGG16', 'IncV3', or 'Res50'
gpu_id          = "0"
num_genes       = 19          # number of classes
date 			      = '20200512'

hdf5_path       = './' + subtype + '_cna2.hdf5'
experiment_name = subtype + 'CNAMultiLabelDataAug' + model_type


import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))

#import required packages
import h5py
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from talos.model.normalizers import lr_normalizer
from talos.metrics.keras_metrics import f1score, precision, recall

from keras import losses
from keras.optimizers import *
from keras.models import Sequential
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation
from keras.preprocessing.image import ImageDataGenerator

import h5py
import talos as ta
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy.random import seed
from collections import Counter
from sklearn.preprocessing import normalize
from talos.model.normalizers import lr_normalizer
from talos.metrics.keras_metrics import mse, mae, rmse
from keras.losses import binary_crossentropy
from sklearn.preprocessing import StandardScaler


from keras import losses
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation


from scipy.stats import sem
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Set hyperoptimization parameters

param_dict = {'batch_size': [64, 128], 
		          'epochs': [20],
		          'lr': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10],
		          'optimizer': [Adam],
              'early_stopper': ['moderate']}


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


def auc(y_true, y_pred):
  ''' 
  AUC function to serve as metric for evaluation during hyperparameter optimization.
  Computes non-weighted average of AUC values across all classes.
  '''
  auc_sum = 0
  for i in range(n_classes):
    auc_sum += tf.metrics.auc(y_true[:,i], y_pred[:,i])[1]
  avg_auc = tf.math.divide(auc_sum, n_classes)
  return avg_auc


def cna_multi_aug_model(x_train, y_train, x_val, y_val, params):
  '''Input mutation multi-label classification model with data augmentation to be submitted to talos.Scan
  Notes: 
    model_type is a global variable used to specify the base model for transfer learning
    train_generator and val_generator are global variables
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
  x = Dense(num_genes, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=x)
  # train the top added layers first, keep the VGG16 layers frozen
  # we can relax this restriction after the last layer is trained
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                loss=binary_crossentropy, 
                metrics=['accuracy', auc, precision, recall, f1score])
  # Perform Data Augmentation while training
  batch_size = params['batch_size']
  out = model.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size, shuffle=False),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=params['epochs'],
                            validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size, shuffle=True),
                            validation_steps=x_val.shape[0] // batch_size,
                            callbacks=[ta.utils.early_stopper(params['epochs'], mode=params['early_stopper'])])
  return out, model




train_generator = ImageDataGenerator(rescale=1.0/255.0,
	horizontal_flip=True, 
	vertical_flip=True, 
	rotation_range=90)
val_generator = ImageDataGenerator(rescale=1.0/255.0)

scan_object = ta.Scan(x=train_X, y=train_y,
                      x_val=val_X, y_val=val_y,
                      val_split=0,
                      params=param_dict,
                      model=cna_multi_aug_model,
                      experiment_name=experiment_name + '_scan',
                      print_params=True,
                      save_weights=True,
                      random_method='quantum',
                      fraction_limit=0.56)


# Analyze hyperparameter optimization results

print("********Scan RESULTS*********")
print("Hypopt results:\n", scan_object.data)
print("Scan object details:\n", scan_object.details)
print("Time for each permutation:\n", scan_object.round_times)

analyze_object = ta.Analyze(scan_object)
print("********Analyze RESULTS*********")
print("Best hyperparameters:\n", analyze_object.best_params('loss', exclude=[], n=3, ascending=False))


# Save predictions in the training, validation, and test set

predict_batch_size = 32
rescale_generator = ImageDataGenerator(rescale=1.0/255.0)
model = scan_object.best_model(metric='loss', asc=True)

train_predict = model.predict_generator(generator=rescale_generator.flow(x=train_X, batch_size=predict_batch_size, shuffle=False), 
                                        steps=train_X.shape[0]/predict_batch_size, 
                                        verbose=1)
val_predict = model.predict_generator(generator=rescale_generator.flow(x=val_X, batch_size=predict_batch_size, shuffle=False), 
                                      steps=val_X.shape[0]/predict_batch_size,
                                      verbose=1)
test_predict = model.predict_generator(generator=rescale_generator.flow(x=test_X, batch_size=predict_batch_size, shuffle=False), 
                                       steps=test_X.shape[0]/predict_batch_size,
                                       verbose=1)


# Write train, val, and test predictions to file

for i, predictions in enumerate([train_predict, val_predict, test_predict]):
    split = ["train", "val", "test"][i]
    with open(experiment_name + date + '_' + split + 'Predictions.txt', 'w') as predictOutput:
        for i in range(len(predictions)):
            _ = predictOutput.write(f"{' '.join([str(num) for num in predictions[i]])}\n")


for i, trues in enumerate([train_y, val_y, test_y]):
    split = ["train", "val", "test"][i]
    with open(experiment_name + date + '_' + split + 'True.txt', 'w') as trueOutput:
        for i in range(len(trues)):
            _ = trueOutput.write(f"{' '.join([str(num) for num in trues[i]])}\n")



# Deploy best model

scan_object.x = np.zeros(500) # necessary for model to restore properly
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + date + "_loss_hyperopt_models", metric="loss", asc=True)
