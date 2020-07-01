# mutation_multi_classification_hypopt_10Cross.py
# Perform 10-fold cross validation for mutation multi-label classification with image augmentation.
#
# 2020.03.02 Eliana Marostica


# Set parameters

subtype         = 'KIRC'      # one of 'KIRC' (clear cell), 'KIRP' (papillary), or 'KICH' (chromophobe)
model_type      = 'IncV3'     # one of 'VGG16', 'IncV3', or 'Res50'
num_genes       =  6          # number of labels per sample
gpu_id          = '0'

hdf5_path_root  = './' + subtype + '_mutation_10FoldCV'
out_path_root   = subtype + "MutMultiAug310FoldCV" + model_type


#ensure version 2.1.2 is being used
import keras; print(keras.__version__)

### Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.44 
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
from tensorflow.keras.metrics import AUC
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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation


from scipy.stats import sem
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


if model_type == "Res50":
    lrs = [0.1, 0.9, 0.99, 0.999]
elif model_type == "IncV3":
    lrs = [0.1, 0.01, 0.001, 0.0001]
elif model_type == "VGG16":
    lrs = [0.1, 0.01, 0.001, 0.0001]
else:
    raise Exception('Did not select a viable base model. Try VGG16, Res50, or IncV3.')


param_dict = {'batch_size': [32], #define parameters for talos hyperparameter tuning
              'epochs': [15],
              'lr': lrs,
              'optimizer': [Adam, RMSprop]}

for foldI in range(10):
  print("\n\n*************************************************************")
  print("**************************FOLD " + str(foldI) + "*****************************")
  print("*************************************************************\n")
  print("Loading " + subtype + " fold " + str(foldI) + "...")

  ## Get numpy arrays
  hf = h5py.File(hdf5_path_root+"F"+str(foldI)+".hdf5", 'r')
  train_X = hf['train_img'][()]
  train_y = hf['train_labels'][()].astype(int)
  test_X = hf['test_img'][()]
  test_y = hf['test_labels'][()].astype(int)
  print("\nRaw Data Dimensions:")
  print("train_X shape: ", train_X.shape)
  print("train_y shape: ", train_y.shape)
  print(" test_X shape: ", test_X.shape)
  print(" test_y shape: ", test_y.shape)
  print()


  ## Define models

  def mutation_multi_aug_model(x_train, y_train, x_val, y_val, params):
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
                  metrics=['accuracy'])
    batch_size = params["batch_size"]
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    it = datagen.flow(x_train, y_train)
    out = model.fit_generator(it,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              epochs=params['epochs'],
                              validation_data=[x_val, y_val],
                              validation_steps=x_val.shape[0] // batch_size,
                              callbacks=[ta.utils.early_stopper(params['epochs'], min_delta=0.03)]) #monitoring val_loss
    return out, model

  scan_object = ta.Scan(x=train_X, y=train_y,
                        val_split=0.3,
                        params=param_dict,
                        model=mutation_multi_aug_model,
                        experiment_name=out_path_root + "_F"+str(foldI) + '_scan',
                        print_params=True,
                        save_weights=True)

  print("********Scan RESULTS*********")
  print("Hypopt results:\n", scan_object.data)
  print("Scan object details:\n", scan_object.details)
  print("Time for each permutation:\n", scan_object.round_times)

  a = ta.Analyze(scan_object)
  print("********Analyze RESULTS*********")
  print("Best hyperparameters:", a.best_params(metric="val_acc", n=1, exclude=[]))

  # Make predictions on training and test data
  print("**********Predictions***********")
  p = ta.Predict(scan_object, task="multi_label")
  train_predict = p.predict(train_X, metric="val_acc")
  test_predict  = p.predict(test_X, metric="val_acc")

  # Write train and test predictions to file
  for i, predictions in enumerate([train_predict, test_predict]):
      split = ["train", "test"][i]
      with open(out_path_root + "_F"+str(foldI) + '_' + split + 'Predictions.txt', 'w') as predictOutput:
          for i in range(len(predictions)):
              _ = predictOutput.write("{}\n".format(" ".join([str(num) for num in predictions[i]])))

  for i, trues in enumerate([train_y, test_y]):
      split = ["train", "test"][i]
      with open(out_path_root + "_F"+str(foldI) + '_' + split + 'True.txt', 'w') as trueOutput:
          for i in range(len(trues)):
              _ = trueOutput.write("{}\n".format(" ".join([str(num) for num in trues[i]])))

  scan_object.x = np.zeros(500) # necessary for model to restore properly
  scan_object.y = np.zeros(500)
  ta.Deploy(scan_object, out_path_root + "_F"+str(foldI) + '_hyperopt_models', metric="val_acc")

  del scan_object
  del a
  del p
