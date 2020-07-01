# survival_mtlr_10Cross_gradcam.py
# grad-CAM visualization generation for survival 10-fold cross validation
#
# 2020.04.16 Kun-Hsing Yu and Eliana Marostica

import keras; print(keras.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU ID

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

from __future__ import absolute_import
from vis.utils import utils
from keras import activations
from keras.models import load_model

import csv
import sys
import numpy as np
from random import shuffle

import matplotlib.cm as cm
from vis.visualization import visualize_cam
import numpy as np
from scipy.ndimage.interpolation import zoom
from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.backprop_modifiers import get
from vis.utils import utils
from vis.visualization.saliency import *
from vis.visualization.saliency import _find_penultimate_layer

from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

from mtlr_cnn import binSurvival

import scipy.misc
from vis.utils import utils
from matplotlib import pyplot as plt
from random import shuffle
import talos as ta

def readInputFileName(inputFilename, testfold):
    imageFilenames=[]
    trainLabels=[]
    testLabels=[]
    with open(inputFilename,'r') as f:
        reader=csv.reader(f,delimiter=' ')
        for imageFilename,survival,event,fold in reader:
            if fold == str(testfold+1):
                imageFilenames.append(imageFilename)
                testLabels.append([float(survival),float(event)])
            else:
                trainLabels.append([float(survival),float(event)])
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)
    testBinned = binSurvival(trainLabels, testLabels, 2)
    return imageFilenames, testBinned


## redefine functions to fix keras-vis bugs
## redefine keras-vis function visualize_cam_with_losses_
def visualize_cam_with_losses_(input_tensor, losses, seed_input, penultimate_layer, grad_modifier=None):
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)
    grads = grads / (np.max(grads) + K.epsilon())
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))
    #print(weights.shape)
    output_dims=[7, 7]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            #print(w.shape)
            #print(penultimate_output_value.shape)
            #z = w * penultimate_output_value[0, ..., i]
            #print(z.shape())
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]
    heatmap = np.maximum(heatmap, 0)
    input_dims=[224, 224]
    zoom_factor = [i / (j * 1.0) for i, j in iter(zip(input_dims, output_dims))]
    heatmap = zoom(heatmap, zoom_factor)
    return utils.normalize(heatmap)


## redefine keras-vis function visualize_cam_
def visualize_cam_(model, layer_idx, filter_indices, seed_input, penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None):
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)
    penultimate_layer = _find_penultimate_layer(model, layer_idx, penultimate_layer_idx)
    losses = [(ActivationMaximization(model.layers[layer_idx], filter_indices), -1)]
    return visualize_cam_with_losses_(model.input, losses, seed_input, penultimate_layer, grad_modifier)


for foldI in range(10):
    print(f"**********************{foldI}**********************\n")
    model_path        = '/mnt/data1/eliana/KIRCSurvivalMTLRStageI10FoldCVRes50_F' + str(foldI) + '_hyperopt_models.zip'
    pred_file_name    = '/mnt/data1/eliana/KIRCSurvivalMTLRStageI10FoldCVRes50_F' + str(foldI) + '_testPredictions.txt'
    testInputFilename = '/mnt/data1/eliana/survival_KIRC_stageI_trainTest10FoldCV.txt'
    subtype           = 'KIRC'
    eventLongPredFilename  = '/mnt/data1/eliana/KIRCSurvivalMTLRStageI10FoldCVRes50_F' + str(foldI) + '_EventLongPredictions.txt' # this script creates this file!
    eventShortPredFilename = '/mnt/data1/eliana/KIRCSurvivalMTLRStageI10FoldCVRes50_F' + str(foldI) + '_EventShortPredictions.txt' # this script creates this file!
    modelname         = "Res50"
    outdir            = "/mnt/data1/eliana/RCCgradCAMSurvival/"
    image_dir_path = '/mnt/data1/tcga' + subtype + '/tissueImages/1000dense200/'
    testImageFilenames, trueTestBinned = readInputFileName(testInputFilename, foldI)
    preds = np.loadtxt(pred_file_name)
    preds_classes = np.round(preds)
    indEventLongSurv  = np.where(trueTestBinned[:,0] == 0)[0].astype(int)
    indEventShortSurv = np.where(trueTestBinned[:,1] == 0)[0].astype(int)
    # Only looking at true positives and true negatives
    # filter filenames so that only true positive and true negative
    indEventLongSurvAccurate  = [i for i in indEventLongSurv if np.array_equal(trueTestBinned[i], preds_classes[i])]
    indEventShortSurvAccurate = [i for i in indEventShortSurv if np.array_equal(trueTestBinned[i], preds_classes[i])]
    filenameEventLongSurv  = [filename for i, filename in enumerate(testImageFilenames) if i in indEventLongSurvAccurate]
    filenameEventShortSurv = [filename for i, filename in enumerate(testImageFilenames) if i in indEventShortSurvAccurate]
    with open(eventLongPredFilename,'w+') as f:
        f.write("\n".join(filenameEventLongSurv))
    with open(eventShortPredFilename,'w+') as f:
        f.write("\n".join(filenameEventShortSurv))
    shortRatio = len(filenameEventShortSurv)/(len(filenameEventShortSurv) + len(filenameEventLongSurv))
    longRatio = len(filenameEventLongSurv)/(len(filenameEventShortSurv) + len(filenameEventLongSurv))
    model = ta.Restore(model_path).model
    #model.summary()
    with open(eventLongPredFilename,'r') as f:
        lines = f.readlines()
        shuffle(lines)
        for i,line in enumerate(lines):
            if i < longRatio*12:
                filename = line.rstrip()
                print(filename)
                image_path = image_dir_path + filename
                img1 = utils.load_img(image_path, target_size=(224, 224))
                if subtype == "CPTAC":
                    output_image_name = outdir + subtype + '_Long_F' + str(foldI) + '_' + modelname + '_' + filename[0:13] + '_' + filename[-14:-4] + '_'
                else:
                    output_image_name = outdir + subtype + '_Long_F' + str(foldI) + '_' + modelname + '_' + filename[0:32] + '_' + filename[-14:-4] + '_'
                ## get the dense layer
                layer_idx = utils.find_layer_idx(model, 'bn5c_branch2c')
                ## to generate gradient-weighted class activation maps
                for modifier in [None, 'guided', 'relu']:
                    for i, img in enumerate([img1]):    
                        img = img.transpose((1,2,0))
                        grads = visualize_cam_(model, layer_idx, filter_indices=0, seed_input=img, backprop_modifier=modifier)        
                        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
                        plt.imsave(output_image_name + 'gradCAM_'+str(i)+'_'+str(modifier)+'.jpg', overlay(jet_heatmap, img.transpose((2,0,1))), cmap='jet')
    with open(eventShortPredFilename,'r') as f:
        lines = f.readlines()
        shuffle(lines)
        for i,line in enumerate(lines):
            if i < shortRatio*12:
                filename = line.rstrip()
                print(filename)
                image_path = image_dir_path + filename
                img1 = utils.load_img(image_path, target_size=(224, 224))
                if subtype == "CPTAC":
                    output_image_name = outdir + subtype + '_Short_F' + str(foldI) + '_' + modelname + '_' + filename[0:13] + '_' + filename[-14:-4] + '_'
                else:
                    output_image_name = outdir + subtype + '_Short_F' + str(foldI) + '_' + modelname + '_' + filename[0:32] + '_' + filename[-14:-4] + '_'
                ## get the dense layer
                layer_idx = utils.find_layer_idx(model, 'bn5c_branch2c')
                ## to generate gradient-weighted class activation maps
                for modifier in [None, 'guided', 'relu']:
                    for i, img in enumerate([img1]):    
                        img = img.transpose((1,2,0))
                        grads = visualize_cam_(model, layer_idx, filter_indices=0, seed_input=img, backprop_modifier=modifier)        
                        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
                        plt.imsave(output_image_name + 'gradCAM_'+str(i)+'_'+str(modifier)+'.jpg', overlay(jet_heatmap, img.transpose((2,0,1))), cmap='jet')


