# mutcount_activation.py
# Visualize activation of output layer for mutation count model
#
# 2020.06.15. Eliana Marostica


from vis.regularizers import TotalVariation, LPNorm
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from __future__ import absolute_import
from vis.utils import utils
from keras import activations
from keras.models import load_model

import csv
import sys
import numpy as np
import random
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

import scipy.misc
from vis.utils import utils
from matplotlib import pyplot as plt
from random import shuffle


# Modify visualize_activation for model with ambiguous input shape

def visualize_activation_with_losses_(input_tensor, losses, wrt_tensor=None,
                                     seed_input=None, input_range=(0, 255),
                                     **optimizer_params):
    """Generates the `input_tensor` that minimizes the weighted `losses`. This function is intended for advanced
    use cases where a custom loss is desired.
    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        losses: List of ([Loss](vis.losses.md#Loss), weight) tuples.
        seed_input: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md#optimizerminimize). Will default to
            reasonable values when required keys are not found.
    Returns:
        The model input that minimizes the weighted `losses`.
    """
    # Default optimizer kwargs.
    optimizer_params = utils.add_defaults_to_kwargs({
        'seed_input': seed_input,
        'max_iter': 200,
        'verbose': False
    }, **optimizer_params)
    opt = Optimizer(tf.ensure_shape(input_tensor, (None, 224, 224, 3)), losses, input_range, wrt_tensor=wrt_tensor, norm_grads=False)
    img = opt.minimize(**optimizer_params)[0]
    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')
    if K.image_data_format() == 'channels_first':
        img = np.moveaxis(img, 0, -1)
    return img


def visualize_activation_(model, layer_idx, filter_indices=None, wrt_tensor=None,
                         seed_input=None, input_range=(0, 255),
                         backprop_modifier=None, grad_modifier=None, lp_norm_weight=10, tv_weight=10,
                         **optimizer_params):
    """Generates the model input that maximizes the output of all `filter_indices` in the given `layer_idx`.
    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils.md#apply_modifications) for better results.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        seed_input: Seeds the optimization with a starting input. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md#optimizerminimize). Will default to
            reasonable values when required keys are not found.
    Example:
        If you wanted to visualize the input image that would maximize the output index 22, say on
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer_idx = dense_layer_idx`.
        If `filter_indices = [22, 23]`, then it should generate an input image that shows features of both classes.
    Returns:
        The model input that maximizes the output of `filter_indices` in the given `layer_idx`.
    """    
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)
    #model.input.set_shape([None, 224, 224, 3])    
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1),
        (LPNorm(tf.ensure_shape(model.input, (None, 224, 224, 3))), lp_norm_weight),
        (TotalVariation(tf.ensure_shape(model.input, (None, 224, 224, 3))), tv_weight)    
    ]    
    # Add grad_filter to optimizer_params.    
    optimizer_params = utils.add_defaults_to_kwargs({
        'grad_modifier': grad_modifier    
    }, **optimizer_params)    
    return visualize_activation_with_losses_(model.input, losses, wrt_tensor,
        seed_input, input_range, **optimizer_params)


################################


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
import gradcam
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

layer_name 		= 'dense_2'
predOutFilename = './KIRCPanCancerMutCountRegressionRes5020200609_2_RankedPredictions.txt' # This file contains the correctly ranked patients' patch filenames (all 4 patients happend to have low-medium mutation count values) (see Interactive Session Code section above for how this file was created)
outdir          = "./RCCDenseVisTumorMutCountRegressionRes50_" + layer_name + "/" # out directory to contain new visualizations (will be created below if it doesn't exist already)
subtype         = 'KIRC'
modelname       = "Res50"
image_dir_path  = '/mnt/data1/tcga' + subtype + '/tissueImages/1000dense200/' # location of pure image patches 
model_path      = './KIRCPanCancerMutCountRegressionRes5020200609_2_hyperopt_models.zip' # talos model path


# Create the outdir if it doesn't already exist
try:
    os.mkdir(outdir)
except OSError:
    print ("Creation of the directory %s failed" % outdir)
else:
    print ("Successfully created the directory %s " % outdir)

from vis.utils import utils
from keras import activations
import talos as ta

# Restore the talos model used
model = ta.Restore(model_path).model


from vis.utils import utils
from vis.visualization import visualize_activation
from keras import activations
import random
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter

inputFile = predOutFilename # for predictions whose ranking was correct (mostly lower mutation counts)
shuffle = True
num_images = 20
output_image_root = outdir + subtype + '_' + modelname + '_'


with open(inputFile,'r') as f:
    lines = f.readlines()
    if shuffle:
        random.shuffle(lines)
    for i,line in enumerate(lines):
        if i <= num_images:
            filename = line.rstrip()
            print(filename)
            image_path = image_dir_path + filename
            img1 = utils.load_img(image_path, target_size=(224, 224))
            output_image_name = output_image_root + filename[0:32] + '_' + filename[-14:-4] + '_'
            ## get the dense layer
            layer_idx = utils.find_layer_idx(model, layer_name)
            ## to generate gradient-weighted class activation maps
            for bp_modifier in [None, 'guided', 'relu']:
                print(f"\t{bp_modifier}")
                for gr_modifier in ['negate']:
                    for i, img in enumerate([img1]):
                        #img = img.transpose((1,2,0))    
                        grads = visualize_activation_(model, layer_idx, seed_input=img, filter_indices=0, backprop_modifier=bp_modifier, grad_modifier=gr_modifier, input_modifiers=[Jitter(16)])
                        img_t = img.transpose((2,0,1))
                        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
                        overlay_img = overlay(img.transpose(2,0,1), jet_heatmap.mean(axis=2))
                        plt.imsave(output_image_name + 'DenseVis_ColorAvg_'+str(bp_modifier)+'_'+str(gr_modifier)+'.jpg', overlay_img, cmap='jet')


# For the high predictions whose rankings were close:

predOutFilename = './KIRCPanCancerMutCountRegressionRes5020200609_2_HighPredictions.txt' # file that contains patch filenames of patients with higher predictions

inputFile = predOutFilename
shuffle = True
num_images = 30
output_image_root = outdir + subtype + '_' + modelname + '_'


with open(inputFile,'r') as f:
    lines = f.readlines()
    if shuffle:
        random.shuffle(lines)
    for i,line in enumerate(lines):
        if i <= num_images:
            filename = line.rstrip()
            print(filename)
            image_path = image_dir_path + filename
            img1 = utils.load_img(image_path, target_size=(224, 224))
            output_image_name = output_image_root + filename[0:32] + '_' + filename[-14:-4] + '_'
            ## get the dense layer
            layer_idx = utils.find_layer_idx(model, layer_name)
            ## to generate gradient-weighted class activation maps
            for bp_modifier in [None, 'guided', 'relu']:
                print(f"\t{bp_modifier}")
                for gr_modifier in [None]: # can also try 'negate' here: see Regression Output Dense layer Visualization section at https://raghakot.github.io/keras-vis/visualizations/activation_maximization/#what-is-activation-maximization
                    for i, img in enumerate([img1]):
                        img = img.transpose((1,2,0))    
                        grads = visualize_activation_(model, layer_idx, seed_input=img, filter_indices=0, backprop_modifier=bp_modifier, grad_modifier=gr_modifier, input_modifiers=[Jitter(16)])
                        img_t = img.transpose((2,0,1))
                        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
                        overlay_img = overlay(img.transpose(2,0,1), jet_heatmap.mean(axis=2))
                        plt.imsave(output_image_name + 'DenseVis_ColorAvg_'+str(bp_modifier)+'_'+str(gr_modifier)+'.jpg', overlay_img, cmap='jet')
