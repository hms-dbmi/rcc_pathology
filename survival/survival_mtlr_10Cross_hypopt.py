## kFoldCVRegressionHDF5.py
#
# store images in an HDF5 database and perform regression using deep learning
# with k-fold cross-validation
#
# with the help of Keras documentation and extensive googling/stackoverflow
#
# Kun-Hsing Yu
# 2017.12.12, simplified 2018.4.5

## Get command-line arguments

import sys

if len(sys.argv) != 4:
    print('Incorrect number of arguments.\n',
          'Please provide subtype, model type, and gpu_id.')
    sys.exit(1)

else:
    subtype     = sys.argv[1]
    model_type  = sys.argv[2]
    gpu_id      = sys.argv[3]


## Set parameters

hdf5_path_root  = './' + subtype + '_survival_10FoldCV'
out_path_root   = subtype + 'SurvivalMTLRStageI10FoldCV' + model_type
nSurvivalBins   = 2

## Ensure that process only uses GPU with id = gpu_id

import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Allowing 100% of GPU memory to be generated to task at hand
set_session(tf.Session(config=config))

import h5py
import tables
import talos as ta
import numpy as np
import pandas as pd
import mtlr_cnn as mtlr
import tensorflow as tf

from numpy.random import seed
from tensorflow import set_random_seed
from lifelines.utils import concordance_index
from talos.model.normalizers import lr_normalizer

from keras import losses
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Masking, Activation


if model_type == "Res50":
    lrs = [0.01, 0.001]
elif model_type == "IncV3":
    lrs = [1e-6, 1e-7]
elif model_type == "VGG16":
    lrs = [1e-5, 1e-6, 1e-7]
else:
    raise Exception('Did not select a viable base model. Try VGG16, Res50, or IncV3.')

param_dict = {'batch_size': [32], # define parameters for talos hyperparameter tuning
              'epochs': [25],
              'lr': lrs,
              'optimizer': [Adam]
              }

for foldI in range(10):
    print("\n\n*************************************************************")
    print("**************************FOLD " + str(foldI) + "*****************************")
    print("*************************************************************\n")
    print("Loading " + subtype + " fold " + str(foldI) + "...")
    
    hf = h5py.File(hdf5_path_root+"F"+str(foldI)+".hdf5", 'r')
    train_X = hf['train_img'][()]
    train_y = hf['train_labels'][()].astype(float)
    test_X = hf['test_img'][()]
    test_y = hf['test_labels'][()].astype(float)
    print("\nRaw Data Dimensions:")
    print("train_X shape: ", train_X.shape)
    print("train_y shape: ", train_y.shape)
    print(" test_X shape: ", test_X.shape)
    print(" test_y shape: ", test_y.shape)
    print()
    
    # Implement upsampling of the training set
    # Indicies of each class' observations
    i_class0 = np.where(train_y[:,1] == 0.)[0]
    i_class1 = np.where(train_y[:,1] == 1.)[0]
    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    # For every observation in class 0, randomly sample from class 1 with replacement
    i_class1_upsampled = np.random.choice(i_class1, size=n_class0, replace=True)
    # Join together class 1's upsampled target vector with class 0's target vector
    train_y_upsampled = np.concatenate((train_y[i_class1_upsampled,:], train_y[i_class0,:]))
    # Join together class 1's upsampled images with class 0's images
    train_X_upsampled = np.concatenate((train_X[i_class1_upsampled,:,:,:], train_X[i_class0,:,:,:]))
    # Save indices and shuffle
    ind = np.concatenate((i_class1_upsampled, i_class0))
    from sklearn.utils import shuffle
    train_X_upsampled, train_y_upsampled, ind = shuffle(train_X_upsampled, train_y_upsampled, ind, random_state=0)
    
    # Bin the y labels
    train_y_binned = mtlr.binSurvival(train_y, train_y_upsampled, nSurvivalBins)     
    test_y_binned = mtlr.binSurvival(train_y, test_y, nSurvivalBins)
    
    print("\nData Dimensions (after upsampling and binning):")
    print("train_X_upsampled shape: ", train_X_upsampled.shape)
    print("   train_y_binned shape: ", train_y_binned.shape)
    print("           test_X shape: ", test_X.shape)
    print("    test_y_binned shape: ", test_y_binned.shape)
    print()
    
    # (Re)define talos model function for each fold
    def ta_model(x_train, y_train, x_val, y_val, params):
        # create model
        if model_type == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False)
        elif model_type == "Res50":
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif model_type == "IncV3":
            base_model = InceptionV3(weights='imagenet', include_top=False)
        else:
            raise Exception('Did not select a viable base model. Try VGG16, Res50, IncV3, Xcept, or InRes.')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(nSurvivalBins, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
        # keep the hidden layers frozen
        for layer in base_model.layers:
            layer.trainable = False
        # compile model with optimizer, loss, and accuracy
        model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), 
                    loss=mtlr.loss, 
                    metrics=[mtlr.accuracy, "categorical_crossentropy"])
        callbacks = [ta.utils.early_stopper(params['epochs'], monitor="val_loss", patience=4)]
        # train the model
        out = model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params["epochs"],
                      callbacks=callbacks,
                      verbose=1,
                      validation_data=[x_val, y_val])
        return out, model
    
    
    scan_object = ta.Scan(x=train_X_upsampled, y=train_y_binned,
                          val_split=0.3,
                          params=param_dict,
                          model=ta_model,
                          experiment_name=out_path_root + "_F"+str(foldI) + '_scan',
                          print_params=True,
                          save_weights=True)
    
    print("********Scan RESULTS*********")
    print("Best model index:", scan_object.best_model(metric='val_loss', asc=True))
    print("Hypopt results:\n", scan_object.data)
    print("Scan object details:\n", scan_object.details)
    print("Time for each permutation:\n", scan_object.round_times)
    print("Round history:\n", scan_object.round_history)
    
    a = ta.Analyze(scan_object)
    print("********Analyze RESULTS*********")
    print("Best hyperparameters:", a.best_params(metric="val_loss", ascending=True, n=1, exclude=[]))
    
    # Make predictions on training and test data
    print("**********Predictions***********")
    p = ta.Predict(scan_object, task="multi_class")
    train_predict = p.predict(train_X, metric="val_loss", asc=True)
    test_predict = p.predict(test_X, metric="val_loss", asc=True)
    
    # Write train and test predictions to file
    for i, predictions in enumerate([train_predict, test_predict]):
        split = ["train","test"][i]
        with open(out_path_root + "_F"+str(foldI) + '_' + split + 'Predictions.txt', 'w') as predictOutput:
            for i in range(len(predictions)):
                _ = predictOutput.write("{}\n".format(" ".join([str(num) for num in predictions[i]])))
    
    scan_object.x = np.zeros(500) # necessary for model to restore properly
    scan_object.y = np.zeros(500)
    ta.Deploy(scan_object, out_path_root + "_F"+str(foldI) + '_hyperopt_models', metric="val_loss", asc=True)
    
    del scan_object
    del a
    del p

