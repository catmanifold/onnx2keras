import logging

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import SlicingOpLambda, Lambda
from typing import Union

from .utils import is_numpy, ensure_tf_type, unsqueeze_tensors_of_rank_one


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
    is_unknown_tensor = input_0.shape == None
    if not is_unknown_tensor and (
            not K.is_keras_tensor(input_0) or not any([input_0.shape[i] == None for i in range(len(input_0.shape))])):
        shapes = []
        for i in input_0.shape:
            if i is not None:
                shapes.append(i)
            else:
                shapes.append(None)
        layers[node_name] = np.array(shapes)
    else:
        layers[node_name] = tf.shape(input_0, out_type=tf.int64)


def optimize_constant_array_for_serialization(params: tf.Tensor, indices: Union[np.ndarray, tf.Tensor], logger):
    """
    Some models contain repetition in the constant array tf.gather gathers from.
    In such cases, serialization takes a long time. serializing 3000 elements take ~3 sec (and scale linearly).
    To optimize for this case, we detect the repetition, take the modulu of the repetition from indices,
    and making the param array shorter.
    This is especially useful for models containing pre-top-k elements.
    Args:
        params: the array to gather on
        indices: the indices to gather

    Returns:
        params: the new array to gather on
        indices: the new indices to gather
    """
    # This Optimization is a must for models with Pre-Top-K needs
    logger.debug('onnx2keras.gather - Encountered long gather. '
                 'Trying to shorten it for easy serialization')
    max_inp = tf.reduce_max(params)
    min_inp = tf.reduce_min(params)
    if len(params) % (max_inp - min_inp + 1) == 0:
        range_inp = tf.range(min_inp, max_inp + 1, dtype=params.dtype)
        reshape_columns = tf.reshape(params, [-1, max_inp + 1])
        if tf.reduce_sum(tf.abs((reshape_columns - range_inp[None, :]))) == 0:
            # [0,1,2,3,0,1,2,3]
            logger.debug('onnx2keras.gather - Shortening sequence - columns')
            indices = indices % (max_inp + 1)
            params = range_inp
        else:
            repetition_len = np.argmin(params == params[0])
            if repetition_len > 0 and len(params) % repetition_len == 0:
                reshaped_rows = tf.reshape(params, [-1, repetition_len])
                first_row = reshaped_rows[:, 0]
                if tf.reduce_sum(tf.abs(reshaped_rows - first_row[:, None])) == 0:
                    logger.debug('onnx2keras.gather - Shortening sequence - rows')
                    indices = indices % repetition_len
                    params = first_row
    return params, indices

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
    OPTIMIZE_ARRAY_LENGTH = 50000
    axis = params.get('axis', 0)
    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]) and not 'is_embedding' in params:
        logger.debug('Gather from numpy array')
        if axis == 0:
            gathered = np.array(layers[node.input[0]][layers[node.input[1]]])
        elif axis == 1:
            gathered = np.array(layers[:, node.input[0]][layers[node.input[1]]])
        elif axis == 2:
            gathered = np.array(layers[:, :, node.input[0]][layers[node.input[1]]])
        elif axis == 3:
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
        if isinstance(layers[node.input[1]], (np.integer, int)) or \
            (not isinstance(layers[node.input[1]], np.ndarray) and \
                K.is_keras_tensor(layers[node.input[1]])):
            #indices are keras tensor or int
            indices = layers[node.input[1]]
        else: #indices are numpy/tf.eager
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
            if tf.is_tensor(indices) and indices.dtype not in [tf.int16, tf.int32, tf.int64]:
                indices = tf.cast(indices, tf.int32)
            if isinstance(indices, list):
                indices = np.array(indices)
            if type(indices) == int:
                out_type = tf.int32
            else:
                out_type = indices.dtype
            dim_len = tf.shape(input_0, out_type=out_type)[axis]  # support None
            if isinstance(indices, (int, np.integer)) and indices < 0:
                indices += dim_len
            if tf.is_tensor(indices):
                indices = tf.where(indices < 0, indices + dim_len, indices)
            if isinstance(input_0, np.ndarray) or not  K.is_keras_tensor(input_0):
                if len(input_0) > OPTIMIZE_ARRAY_LENGTH:
                    input_0, indices = optimize_constant_array_for_serialization(input_0, indices, logger)
            layers[node_name] = tf.gather(input_0, indices, axis=axis)


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
                             range(len(layer_input))]).all() or any(
                [layer_input[i].shape == None for i in range(len(layer_input))]):
                try:
                    layers[node_name] = tf.concat(layer_input, axis=params['axis'], name=keras_name)
                except Exception as ex:
                    # might be due to type mismatch between different inputs of tf.concat
                    raise

            else:
                layer_input = unsqueeze_tensors_of_rank_one(layer_input, axis=params['axis'])
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

    logger.debug(f'input_0: {str(input_0)}')
    logger.debug(f'input_1: {str(input_1)}')

    if is_numpy(input_1):
        dims_to_set_as_zero = None
        dims_to_keep_unchanged = None
        allow_zero = params.get('allowzero', False)
        contains_zero_dim = np.isin(input_1, 0).any()
        contains_infer_dim = np.isin(input_1, -1).any()
        if allow_zero:
            if contains_infer_dim and contains_zero_dim:
                raise ValueError(
                    "Reshape parameter 'allowzero' is set and reshaping argument contains both '0' dim and '-1'"
                    "which is not allowed"
                    f"node name: {node_name}")
            elif contains_zero_dim:
                dims_to_set_as_zero = np.argwhere(input_1 == 0)
        elif not allow_zero and contains_zero_dim:
            dims_to_keep_unchanged = np.squeeze(np.argwhere(input_1 == 0))

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

                reshape = keras.layers.Reshape(np.int32(input_1[1:]),
                                               name=keras_name)  # keras reshape ignores batch dimension but onnx axis do not
                layers[node_name] = reshape(layers[node_name])

            else:
                input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
                input_0_shape = input_0.shape
                if len(input_0_shape) > 0:
                    first_mismatch = np.argmin(np.array(input_0_shape[:len(input_1)]) == input_1)
                else:  # does not need to reshape dynamicalyy (skip next section)
                    first_mismatch = 0
                if (input_1 == None).any() and (np.array(input_0_shape) == None).any() and len(input_1) < len(
                        input_0_shape) \
                        and input_1[first_mismatch] == -1:  # reshape end
                    end_match_arr = np.array(input_0_shape[-len(input_1):]) == input_1
                    end_idx_match = np.argmax((np.array(input_0_shape[-len(input_1):]) == input_1))
                    end_idx_match = end_idx_match + len(input_0_shape) - len(input_1) if end_idx_match > first_mismatch \
                                                                                         and end_match_arr[
                                                                                             end_idx_match] else len(
                        input_0_shape) + 1
                    tf_shape = tf.shape(input_0)
                    layers[node_name] = tf.reshape(input_0, [*tf_shape[:first_mismatch], -1, *tf_shape[end_idx_match:]])
                else:
                    logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
                    logger.debug('Target shape :')
                    logger.debug(np.int32(input_1[1:]))
                    if len(np.int32(input_1[1:])) == 1 and np.int32(input_1[1:])[0] == -1:
                        if input_0.shape.rank == 1:
                            input_0 = tf.expand_dims(input_0, 0)
                        logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                        flatten = keras.layers.Reshape(target_shape=input_1[1:], name=keras_name)
                        layers[node_name] = flatten(input_0)
                    elif len(input_1) == 1 and input_1[0] == -1:
                        layers[node_name] = tf.reshape(input_0, [-1])
                    else:
                        if len(input_0.shape) == 0 or (
                                input_0.shape[0] != input_1[0] and input_1[0] != 0):  # keras reshape don't work
                            new_shape = input_1.copy()
                            if dims_to_set_as_zero is not None:
                                new_shape[dims_to_set_as_zero] = 0
                            elif dims_to_keep_unchanged is not None:
                                new_shape[dims_to_keep_unchanged] = np.array(input_0.shape)[dims_to_keep_unchanged]
                            layers[node_name] = tf.reshape(input_0, new_shape, name=keras_name)
                        else:
                            reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                            layers[node_name] = reshape(input_0)
    else:  # dynamic reshape
        layers[node_name] = tf.reshape(input_0, input_1)


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

    if len(np.unique(params['axes'])) < len(params['axes']):
        raise AttributeError(f"The specified axes contains duplicates values: {params['axes']}")

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        layers[node_name] = layers[node.input[0]]
        for axis in params['axes']:
            layers[node_name] = np.expand_dims(layers[node_name], axis)
    else:
        unsqueezed_input = layers[node.input[0]]
        for axis in params['axes']:
            unsqueezed_input = tf.expand_dims(unsqueezed_input, axis)

        layers[node_name] = unsqueezed_input


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
    input_dims = tf.shape(input_0)
    flatten_axis = params.get('axis', 1)
    reshaped_input = tf.reshape(input_0, [tf.reduce_prod(input_dims[:flatten_axis]),
                                          tf.reduce_prod(input_dims[flatten_axis:])])
    layers[node_name] = reshaped_input


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
        starts = list(layers[node.input[1]])
        ends = list(layers[node.input[2]])
        try:
            axes = list(layers[node.input[3]])
        except:
            input_rank = len(layers[node.input[0]].shape)
            axes = list(range(input_rank))
        try:
            steps = list(layers[node.input[4]])
        except IndexError:
            steps = list(params.get("steps", [None] * len(axes)))
    try:
        max_len = len(layers[node.input[0]].shape)
        axes_positives = [axis if axis >= 0 else max_len + axis for axis in axes]
    except ValueError as e:
        if layers[node.input[0]].shape == None:  # tensor with unknown shape (not the same as dynamic)
            max_len = max(axes) + 1
            if any([axis < 0 for axis in axes]):
                raise NotImplementedError("For a tensor with unknown shape, can't use negative axis")
            else:
                axes_positives = axes
        else:
            raise NotImplementedError(f"Couldn't transform the axis in a slice layer {node_name}")
    slice_spec_param = []
    is_dynamic = False
    for i in range(len(starts)):
        for index_li in [starts, steps, ends]:
            if index_li[i] is not None and not isinstance(index_li[i], int) and not is_numpy(
                    index_li[i]) and K.is_keras_tensor(index_li[i]):
                is_dynamic = True
    if not is_dynamic:
        for axis in range(max_len):
            if axis in axes_positives:
                axis_index = axes_positives.index(axis)
                start = starts[axis_index]
                end = ends[axis_index]
                step = steps[axis_index]
                slice_spec_param.append({'start': start, 'step': step, 'stop': end})
            else:
                slice_spec_param.append({'start': None, 'step': None, 'stop': None})
        if is_numpy(layers[node.input[0]]) and np.array([_shape is None for _shape in layers[node.input[0]]]).any() \
                and len(layers[node.input[0]].shape) == 1:  # slice numpy array which is a shape
            sliced = layers[node.input[0]][start:end:step]
        else:
            input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
            slicing_layer = SlicingOpLambda(tf.__operators__.getitem)
            sliced = slicing_layer(input_0, slice_spec=slice_spec_param)
            if is_numpy(layers[node.input[0]]) and not K.is_keras_tensor(sliced):
                sliced = sliced.numpy()
        layers[node_name] = sliced
    else:
        try:
            steps = list(layers[node.input[4]])
        except IndexError:
            steps = list(params.get("steps", [1] * len(axes)))
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        keras_shape = tf.shape(layers[node.input[0]])
        start_vec = [0] * max_len
        end_vec = [keras_shape[i] for i in range(max_len)]
        step_vec = [1] * max_len
        for axis in range(max_len):
            if axis in axes_positives:
                axis_index = axes_positives.index(axis)
                for res_list, input_list in zip([start_vec, step_vec, end_vec], [starts, steps, ends]):
                    slice_index = input_list[axis_index]
                    if input_list[axis_index] is not None and not isinstance(slice_index, int) and not is_numpy(
                            input_list[axis_index]) and input_list[axis_index].dtype != tf.int32:
                        slice_index = tf.cast(slice_index, tf.int32)
                    res_list[axis] = slice_index
        layers[node_name] = tf.strided_slice(input_0,
                                             tf.concat([start_vec], axis=0),
                                             tf.concat([end_vec], axis=0),
                                             tf.concat([step_vec], axis=0))


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
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    axis = None
    if 'axes' in params:
        axis = params['axes'][0]

    if len(node.input) == 2:
        axis = layers[node.input[1]].tolist()
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

    rank = len(input_tensor.shape)
    axes = params.get('axes', list(range(rank)))  # Default to resizing all axes

    # Validate axes
    axes = [a if a >= 0 else a + rank for a in axes]  # Convert negative axes to positive
    if any(a < 0 or a >= rank for a in axes):  # check that all axes values are within input rank
        raise ValueError("Invalid axes value")

    to_channel_last = keras.layers.Permute((2, 3, 1))(input_tensor)  # (B, W, H, C)
    shape = tf.cast(tf.shape(to_channel_last), tf.int32)
    tf_resize_shapes = [shape[i] for i in range(1, 3)]  # (W, H)

    if len(scales) > 0:
        for i, axis in enumerate(axes):
            if scales[i] != 1:
                tf_resize_shapes[axis - 2] = tf.cast(scales[i] * tf.cast(tf_resize_shapes[axis - 2], tf.float32), tf.int32)
    else:
        for i, axis in enumerate(axes):
            if sizes[i] != input_tensor.shape[axis]:
                tf_resize_shapes[axis - 2] = int(sizes[i])

    resized = tf.image.resize(to_channel_last,
                              size=tf.stack(tf_resize_shapes, axis=0),
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
    input_1 = layers[node.input[1]]
    if input_0.dtype.is_bool:
        input_0 = tf.cast(input_0, dtype='int32')
    multiply_res = input_0 * tf.ones(shape=input_1, dtype=input_0.dtype)
    # input_0.dtype == np.int32 since we can't serialize constants as int64, need to cast to true type
    if layers[node.input[0]].dtype == np.int64:
        multiply_res = tf.cast(multiply_res, tf.int64)
    layers[node_name] = multiply_res


def convert_tile(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.tile(layers[node.input[0]], layers[node.input[1]])


def convert_gather_elements(node, params, layers, lambda_func, node_name, keras_name):
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
    logger = logging.getLogger('onnx2keras.gather_elements')
    axis = params.get('axis', 0)
    data_input = layers[node.input[0]]
    indices_input = layers[node.input[1]]

    def torch_gather(x, indices, gather_axis):

        all_indices = tf.where(tf.fill(indices.shape, True))
        gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

        gather_indices = []
        for axis in range(len(indices.shape)):
            if axis == gather_axis:
                gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
            else:
                gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

        gather_indices = tf.stack(gather_indices, axis=-1)
        gathered = tf.gather_nd(x, gather_indices)
        reshaped = tf.reshape(gathered, indices.shape)
        return reshaped

    layers[node_name] = torch_gather(data_input, indices_input, axis)
