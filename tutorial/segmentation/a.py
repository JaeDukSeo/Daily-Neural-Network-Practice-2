from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = get_kernel_size(factor)
    weights = np.zeros((filter_size, filter_size,
                        number_of_classes,number_of_classes), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    
    return weights

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.append("/home/dpakhom1/workspace/my_models/slim/")
checkpoints_dir = '/home/dpakhom1/checkpoints'

image_filename = 'cat.jpg'
annotation_filename = 'cat_annotation.png'

image_filename_placeholder = tf.placeholder(tf.string)
annotation_filename_placeholder = tf.placeholder(tf.string)
is_training_placeholder = tf.placeholder(tf.bool)

feed_dict_to_use = {image_filename_placeholder: image_filename,
                    annotation_filename_placeholder: annotation_filename,
                    is_training_placeholder: True}

image_tensor = tf.read_file(image_filename_placeholder)
annotation_tensor = tf.read_file(annotation_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
annotation_tensor = tf.image.decode_png(annotation_tensor, channels=1)

# Get ones for each class instead of a number -- we need that
# for cross-entropy loss later on. Sometimes the groundtruth
# masks have values other than 1 and 0. 
class_labels_tensor = tf.equal(annotation_tensor, 1)
background_labels_tensor = tf.not_equal(annotation_tensor, 1)

# Convert the boolean values into floats -- so that
# computations in cross-entropy loss is correct
bit_mask_class = tf.to_float(class_labels_tensor)
bit_mask_background = tf.to_float(background_labels_tensor)

combined_mask = tf.concat(concat_dim=2, values=[bit_mask_class,bit_mask_background])

# Lets reshape our input so that it becomes suitable for 
# tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))














# --- end code ---