# subtype_classifier_hypopt.py
# Run hyperparameter optimization for subtype classification
#
# 2019.09.07 Eliana Marostica

# Set parameters

model_type      = 'Res50'     # one of 'VGG16', 'IncV3', or 'Res50'
date            = '20200628'  # today's date and/or any additional information to make a unique identifier
gpu_id          = '2'


hdf5_path       = './' + subtype + '_bypid.hdf5'        # automatically generated for this task
experiment_name = "Subtype" + model_type  # automatically generated for this task



### Ensure that process only uses X GPU's 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID
os.system('nvidia-smi')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))
os.system('nvidia-smi')

#ensure version 2.1.2 is being used
import keras; print(keras.__version__)

#import required packages
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
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation


# Define parameters for talos hyperparameter tuning

p = {'batch_size': [32], #define parameters for talos hyperparameter tuning
     'epochs': [15],
     'lr': [0.01, 0.001, 0.0001, 0.00001],
     'optimizer': [Adam, RMSprop],
     'losses': ['categorical_crossentropy']}


# Get numpy arrays of data
# data was amalgamated from the malignancy HDF5 files

# get KIRP images and generate one-hot encoded labels (label == 0)
hdf5_path = './KIRP_bypid.hdf5'
hf        = h5py.File(hdf5_path, 'r')
train_X   = hf['train_img'][()]
train_y   = np.repeat(np.array([[1,0,0]]), train_X.shape[0], axis=0) #make one-hot array for multiclassification
val_X     = hf['val_img'][()]
val_y     = np.repeat(np.array([[1,0,0]]), val_X.shape[0], axis=0) 
test_X    = hf['test_img'][()]
test_y    = np.repeat(np.array([[1,0,0]]), test_X.shape[0], axis=0) 

# get KIRC images and generate one-hot encoded labels (label == 1)
hdf5_path = './KIRC_bypid.hdf5'
hf        = h5py.File(hdf5_path, 'r')
train_X   = np.concatenate((train_X, hf['train_img'].value), axis=0)
train_y   = np.concatenate((train_y, np.repeat(np.array([[0,1,0]]), hf['train_img'].value.shape[0], axis=0)), axis=0)
val_X     = np.concatenate((val_X, hf['val_img'].value), axis=0)
val_y     = np.concatenate((val_y, np.repeat(np.array([[0,1,0]]), hf['val_img'].value.shape[0], axis=0)), axis=0)
test_X    = np.concatenate((test_X, hf['test_img'].value), axis=0)
test_y    = np.concatenate((test_y, np.repeat(np.array([[0,1,0]]), hf['test_img'].value.shape[0], axis=0)), axis=0)

# get KICH images and generate one-hot encoded labels (label == 2)
hdf5_path = './KICH_bypid.hdf5'
hf        = h5py.File(hdf5_path, 'r')
train_X   = np.concatenate((train_X, hf['train_img'].value), axis=0)
train_y   = np.concatenate((train_y, np.repeat(np.array([[0,0,1]]), hf['train_img'].value.shape[0], axis=0)), axis=0)
val_X     = np.concatenate((val_X, hf['val_img'].value), axis=0)
val_y     = np.concatenate((val_y, np.repeat(np.array([[0,0,1]]), hf['val_img'].value.shape[0], axis=0)), axis=0)
test_X    = np.concatenate((test_X, hf['test_img'].value), axis=0)
test_y    = np.concatenate((test_y, np.repeat(np.array([[0,0,1]]), hf['test_img'].value.shape[0], axis=0)), axis=0)


# Print dimensions of amalgmated subtype data

print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
print(test_X.shape)
print(test_y.shape)


# Shuffle training, validation, and test sets

np.random.seed(56201)
rng_state = np.random.get_state()
np.random.shuffle(train_X)
np.random.set_state(rng_state)
np.random.shuffle(train_y)

np.random.seed(8187)
rng_state = np.random.get_state()
np.random.shuffle(val_X)
np.random.set_state(rng_state)
np.random.shuffle(val_y)

np.random.seed(18460)
rng_state = np.random.get_state()
np.random.shuffle(test_X)
np.random.set_state(rng_state)
np.random.shuffle(test_y)



# Define methods

def save_labels(labels, filename):
  '''Save binary predictions or true values to text files.
  Args:
    labels : list of training, val, and test labels
  '''
  for i, labs in enumerate(labels):
    split = ["train", "val", "test"][i]
    with open(f"{filename}_{split}.txt", 'w') as fileOutput:
        for i in range(len(labs)):
            _ = fileOutput.write(f"{labs.flatten()[i]}\n")


def subtype_model(x_train, y_train, x_val, y_val, params):
    '''Input subtype model to be submitted to talos.Scan
    Notes: model_type is a global variable used to specify the base model for transfer learning
    '''
    if model_type == "VGG16":   # model_type is a global variable
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
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    # train the top added layers first, keep the VGG16 layers frozen
    # we can relax this restriction after the last layer is trained
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
                      model=subtype_model,
                      experiment_name=experiment_name + '_scan/')


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

# heatmap correlation
plt.clf()
analyze_object.plot_corr('val_acc', ['val_loss', 'acc', 'loss'])
plt.savefig(experiment_name + '_scan/' + date + '_corr.png')

# a four dimensional bar grid
plt.clf()
analyze_object.plot_bars('lr', 'val_acc', 'kernel_initializer', 'bias_initializer')
plt.savefig(experiment_name + '_scan/' + date + '_bar_grid.png')



# Save predictions in the training, validation, and test set

p = ta.Predict(scan_object, task='multi_class')
train_predict = p.predict(train_X, metric="val_acc", asc=False)
val_predict   = p.predict(val_X, metric="val_acc", asc=False)
test_predict  = p.predict(test_X, metric="val_acc", asc=False)


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

scan_object.x = np.zeros(500)
scan_object.y = np.zeros(500)
ta.Deploy(scan_object, experiment_name + date + '_hyperopt_models', metric="val_acc", asc=False)

