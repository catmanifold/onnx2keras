import logging

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import SlicingOpLambda

from .utils import is_numpy, ensure_tf_type, ensure_numpy_type


def convert_transpose(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert transpose.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.transpose')
    input_name = node.input[0]

    if params['perm'][0] != 0:
        logger.warning('Can\'t permute batch dimension. Result may be wrong.')
        if is_numpy(layers[input_name]):
            logger.warning('Transposing numpy array.')
            layers[node_name] = np.transpose(layers[input_name], axes=params['perm'])
        else:
            layers[node_name] = tf.transpose(layers[input_name], perm=params['perm'])
    else:
        permute = keras.layers.Permute(params['perm'][1:], name=keras_name)
        layers[node_name] = permute(layers[input_name])


def convert_shape(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert shape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.shape')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    logger.debug('Actual shape:')
    logger.debug(np.array(input_0.shape))

    shapes = []
    for i in input_0.shape:
        if i is not None:
            shapes.append(i)
        else:
            shapes.append(None)

    layers[node_name] = np.array(shapes)


def convert_gather(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert gather.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.gather')

    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]) and not 'is_embedding' in params:
        logger.debug('Gather from numpy array')

        if params['axis'] == 0:
            gathered = np.array(layers[node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 1:
            gathered = np.array(layers[:, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 2:
            gathered = np.array(layers[:, :, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 3:
            gathered = np.array(layers[:, :, :, node.input[0]][layers[node.input[1]]])
        else:
            raise AttributeError('Can\'t gather by axis more than 3.')

        if gathered.dtype == np.object0:
            try:
                gathered = gathered.astype(np.int32)
            except TypeError:
                pass
        layers[node_name] = gathered
    else:
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        if not isinstance(layers[node.input[1]], np.ndarray) and \
                K.is_keras_tensor(layers[node.input[1]]):
            indices = layers[node.input[1]]
        else:
            indices = layers[node.input[1]]
            if not is_numpy(layers[node.input[1]]):
                indices = indices.numpy()
            indices = indices.tolist()
        if "is_embedding" in params:
            if len(input_0.shape) == 2:
                emb = tf.keras.layers.Embedding(input_0.shape[0], input_0.shape[1], weights=[layers[node.input[0]]],
                                                name=keras_name)
                if isinstance(indices, list):
                    layers[node_name] = emb(np.array(indices))
                else:
                    layers[node_name] = emb(indices)
            else:
                raise AttributeError("Cannot transform gather into embedding with non 2D array")
        else:
            layers[node_name] = tf.gather(input_0, indices, axis=params['axis'])


def convert_concat(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert concat.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.concat')

    layer_input = [layers[node.input[i]] for i in range(len(node.input))]

    if all([is_numpy(layers[node.input[i]]) for i in range(len(node.input))]):
        logger.debug('Concat numpy arrays.')
        layers[node_name] = np.concatenate(layer_input, axis=params['axis'])
    else:
        logger.debug('Concat Keras layers.')
        if len(layer_input) > 1:
            if not np.array([tf.is_tensor(layer_input[i]) and K.is_keras_tensor(layer_input[i]) for i in
                             range(len(layer_input))]).all():
                try:
                    layers[node_name] = tf.concat(layer_input, axis=params['axis'], name=keras_name)
                except Exception as ex:
                    # might be due to type mismatch between different inputs of tf.concat
                    raise

            else:
                layers[node_name] = keras.layers.concatenate(inputs=layer_input,
                                                             axis=params['axis'],
                                                             name=keras_name)
        else:
            layers[node_name] = layer_input[0]


def convert_reshape(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reshape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.reshape')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    if is_numpy(input_1):
        logger.debug('The second argument is numpy array.')
        if is_numpy(input_0):
            logger.debug('The first argument is numpy array. Apply np.reshape.')
            layers[node_name] = np.reshape(input_0, np.int32(input_1))
        else:
            if params['change_ordering']:
                input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

                # Fix critical issue with NHWC
                if input_1[0] is None and input_1[1] == -1:
                    logger.warning('!!! IMPORTANT INFORMATION !!!')
                    logger.warning('The target shape if [None, -1] that means flatten.')
                    logger.warning('But the target ordering is NHWC, so we cant simply perform flatten')
                    logger.warning('The layer will be converted as lambda with tf.transpose')
                    logger.warning('---')

                    def target_layer(x):
                        import tensorflow as tf
                        x = tf.transpose(x, [0, 3, 1, 2])
                        return x

                    lambda_layer = keras.layers.Lambda(target_layer, name="%s_CHW" % keras_name)
                    layers[node_name] = lambda_layer(input_0)
                    lambda_func[keras_name] = target_layer
                else:
                    layers[node_name] = input_0

                reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                layers[node_name] = reshape(layers[node_name])

            else:
                input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
                input_0_shape = input_0.shape
                first_mismatch = np.argmin(np.array(input_0_shape[:len(input_1)]) == input_1)
                if (input_1 == None).any() and (np.array(input_0_shape) == None).any() and len(input_1) < len(input_0_shape)\
                        and input_1[first_mismatch] == -1: #reshape end
                    end_match_arr = np.array(input_0_shape[-len(input_1):]) == input_1
                    end_idx_match = np.argmax((np.array(input_0_shape[-len(input_1):]) == input_1))
                    end_idx_match = end_idx_match + len(input_0_shape) - len(input_1) if end_idx_match > first_mismatch \
                                                     and end_match_arr[end_idx_match] else len(input_0_shape) + 1
                    tf_shape = tf.shape(input_0)
                    layers[node_name] = tf.reshape(input_0, [*tf_shape[:first_mismatch], -1, *tf_shape[end_idx_match:]])
                else:
                    logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
                    logger.debug('Target shape :')
                    logger.debug(np.int32(input_1[1:]))

                    if len(np.int32(input_1[1:])) == 1 and np.int32(input_1[1:])[0] == -1:
                        logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                        flatten = keras.layers.Flatten(name=keras_name)
                        layers[node_name] = flatten(input_0)
                    else:
                        if input_0.shape[0] != input_1[0]:  # keras reshape don't work
                            layers[node_name] = tf.reshape(input_0, input_1, name=keras_name)
                        else:
                            reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                            layers[node_name] = reshape(input_0)
    else:
        raise AttributeError('Can\'t reshape dynamic size.')


def convert_unsqueeze(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert unsqueeze.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.unsqueeze')

    if len(node.input) != 1:
        if len(node.input) == 2:
            params['axes'] = layers[node.input[1]]
        else:
            raise AttributeError('Number of inputs is not equal 1 for unsqueeze layer')

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        layers[node_name] = layers[node.input[0]]
        for axis in params['axes']:
            layers[node_name] = np.expand_dims(layers[node_name], axis)
    else:

        if len(params['axes']) != 1:
            raise AttributeError('Number of axes is not equal 1. Cannot unsqueeze')

        layers[node_name] = tf.expand_dims(layers[node.input[0]], params['axes'][0])


def convert_flatten(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert flatten.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.flatten')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for flatten layer')

    logger.debug('Convert inputs to Keras/TF layers if needed.')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    # Fix critical issue with flatten
    permute = keras.layers.Permute((3, 1, 2))
    tensor_chw = permute(input_0)
    flatten = keras.layers.Flatten(name=keras_name)
    layers[node_name] = flatten(tensor_chw)


def convert_slice(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert slice.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.slice')

    if params['change_ordering']:
        raise NotImplementedError("change_ordering for Slice is not implemented")
    if 'axes' in params:
        axes = list(params["axes"])
        ends = list(params["ends"])
        starts = list(params["starts"])
        steps = list(params.get("steps", [None] * len(axes)))
    else:
        starts = list(ensure_numpy_type(layers[node.input[1]]))
        ends = list(ensure_numpy_type(layers[node.input[2]]))
        axes = list(ensure_numpy_type(layers[node.input[3]]))
        try:
            steps = list(ensure_numpy_type(layers[node.input[4]]))
        except IndexError:
            steps = list(params.get("steps", [None] * len(axes)))

    input_shape_len = len(layers[node.input[0]].shape)
    axes_positives = [axis if axis >= 0 else input_shape_len + axis for axis in axes]

    slice_spec_param = []
    for axis in range(input_shape_len):
        if axis in axes_positives:
            axis_index = axes_positives.index(axis)
            start = starts[axis_index]
            end = ends[axis_index] if ends[axis_index] < 2147483647 else None
            step = steps[axis_index]
            slice_spec_param.append({'start': start, 'step': step, 'stop': end})
        else:
            slice_spec_param.append({'start': None, 'step': None, 'stop': None})
    if is_numpy(layers[node.input[0]]) and np.array([_shape is None for _shape in layers[node.input[0]]]).any()\
            and len(layers[node.input[0]].shape) == 1: # slice numpy array which is a shape
        sliced = layers[node.input[0]][start:end:step]
    else:
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        slicing_layer = SlicingOpLambda(tf.__operators__.getitem)
        sliced = slicing_layer(input_0, slice_spec=slice_spec_param)
        if is_numpy(layers[node.input[0]]):
            sliced = sliced.numpy()
    layers[node_name] = sliced


def convert_squeeze(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Squeeze layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for squeeze layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    axis = None
    if 'axes' in params:
        axis = params['axes'][0]
    layers[node_name] = tf.squeeze(input_0, axis=axis)


def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
    logger = logging.getLogger('onnx2keras.reshape')

    input_tensor = layers[node.input[0]]
    roi = None if len(node.input[1]) == 0 else layers[node.input[1]]
    scales = [] if len(node.input[2]) == 0 else layers[node.input[2]]
    sizes = None
    if len(node.input) == 4:
        sizes = layers[node.input[3]]
    if roi:
        raise Exception("Resize with roi not supported")

    if params['mode'] == b'nearest':
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif params['mode'] == b'cubic':
        resize_method = tf.image.ResizeMethod.BICUBIC
    elif params['mode'] == b'linear':
        resize_method = tf.image.ResizeMethod.BILINEAR
    else:
        raise Exception("unsupported resize method")

    to_channel_last = keras.layers.Permute((2, 3, 1))(input_tensor)
    if len(scales) > 0:
        if scales[0] != 1 or scales[1] != 1:
            raise Exception("Resize of channels or batch dim not suppported")

        tf_resize_shapes = [int(scales[2] * to_channel_last.shape[1]),
                            int(scales[3] * to_channel_last.shape[2])]
    else:
        if sizes[0] != input_tensor.shape[0] or sizes[1] != input_tensor.shape[1]:
            raise Exception("Resize of channels or batch dim not suppported")
        tf_resize_shapes = [int(sizes[2]), int(sizes[3])]

    resized = tf.image.resize(to_channel_last,
                              size=tf_resize_shapes,
                              method=resize_method)
    to_channel_first = keras.layers.Permute((3, 1, 2))(resized)
    layers[node_name] = to_channel_first


def convert_expand(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Expand layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 2:
        assert AttributeError('More than 2 input for expand layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    input_1 = ensure_numpy_type(layers[node.input[1]]).astype(np.int32)

    layers[node_name] = input_0 * tf.ones(input_1, dtype=input_0.dtype)


def convert_tile(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.tile(layers[node.input[0]], layers[node.input[1]])
