# MTLR CNN
# Functions for implementing multi-task linear regression using CNNs for survival prediction.
# Eliana Marostica, January 2020 

from math import ceil
from keras import losses

import numpy as np
import tensorflow as tf
import keras.backend as K


def binSurvival(train_labels, labels, nSurvivalBins, method="equal"):
    '''
    Bin censored survival data into nSurvivalBins. Will throw an error if nSurvivalBins==1. 
    Bins always start at 0 and go up to the maximum survival value present in the training data.
    Bins can be created of equal size (method="equal") or based on quantiles in the training data (method="quantile" or method="quantile_eo").
    
    Params
    ------
    train_labels   : a 2D numpy array of training labels containing time and event data, from which the maximum survival and bins are determined
    labels         : a 2D numpy array of survival labels containing time and event data, to be binned according to the training data
    nSurvivalBins  : the approximate number of survival bins
    method         : the method for creating the bin boundaries; one of "equal", "quantile", or "quantile_eo" for quantile using only event data points 
    
    Returns
    -------
    binnedSurvival : a 2D numpy array of binned survival labels for each sample in the data subset
    '''
    assert nSurvivalBins > 1, "Creating only one bin will result in all samples having the same label."
    assert method in ["equal", "quantile", "quantile_eo"], 'Method for binning must be one of "equal", "quantile", or "quantile_eo"'
    if method == "equal":
        trainMaxSurv = ceil(max(train_labels[:,0]))
        bins = [trainMaxSurv/nSurvivalBins*i for i in range(nSurvivalBins+1)]
    elif method == "quantile_eo":
        events = np.extract(train_labels[:,1] == 1, train_labels[:,0])
        bins = [np.quantile(events, q) for q in np.linspace(0,1,nSurvivalBins+1)]
        bins[0] = 0.
    else:
        bins = [np.quantile(train_labels[:,0], q) for q in np.linspace(0,1,nSurvivalBins+1)]
        bins[0] = 0.
    if np.array_equal(train_labels, labels): 
        print("Number of bins:", len(bins)-1) # print the actual number of bins being used
        print("Bins:\n", bins)
    binnedSurvival = np.zeros((labels.shape[0], len(bins)-1))
    for i in range(labels.shape[0]):
        if (labels[i,1] == 1):
            binnedSurvival[i] = np.array([int(labels[i,0] >= bins[j-1] and labels[i,0] < bins[j]) for j in range(1,len(bins))]) #one-hot encoding for event (non-censored) samples
        else:
            binnedSurvival[i] = np.array([int(labels[i,0] < up) for up in bins[1:]])
        if labels[i,0] >= bins[-1]: binnedSurvival[i,-1] = 1 # If val/test set survival value is greater than max survival in train set, label survival value to be in last bin
    return binnedSurvival


def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multi-task linear regression classification problems.
    '''
    return K.mean(tf.linalg.tensor_diag_part(K.dot(K.cast(y_true, dtype='float32'), K.transpose(K.cast(K.equal(y_true,K.round(y_pred)), dtype='float32')))))

def loss(y_true, y_pred):
	'''Calculates an adapted version of the ordinal_categorical_crossentropy
	from https://github.com/JHart96/keras_ordinal_categorical_crossentropy
	for multi-task linear regression classification problems.
	'''
	weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
	fix = K.cast(tf.logical_not(K.all(K.stack([K.sum(y_true, axis=1) > 1, K.argmax(y_pred, axis=1) >= K.argmax(y_true, axis=1)], axis=0), axis=0)), dtype="float32")
	return (1.0 + weights) * fix * losses.categorical_crossentropy(y_true, y_pred)