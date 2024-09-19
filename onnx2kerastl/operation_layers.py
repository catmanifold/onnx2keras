import logging

import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

from .customonnxlayer.onnxeinsum import OnnxEinsumLayer
from .exceptions import UnsupportedLayer
from .utils import is_numpy, ensure_tf_type, ensure_float
from .tfops_funcs import tf_math_abs, tf_clip_by_value, tf_math_negative, K_mean, tf_math_reduce_prod,\
    tf_math_reduce_min, tf_math_pow, tf_math_sqrt, tf_cast, tf_argmax, tf_expand_dims, tf_math_reciprocal,\
    tf_logical_not, tf_math_sign, tf_math_sin, tf_math_cosh, tf_math_ceil, tf_math_acosh, tf_math_acos,\
    tf_math_asinh, tf_math_asin, tf_math_atanh, tf_math_tan, tf_math_atan, tf_math_sinh, tf_math_less_equal,\
    tf_bitwise_invert, tf_bitwise_bitwise_and, tf_bitwise_bitwise_or, tf_bitwise_bitwise_xor, tf_cos,\
    tf_math_greater, tf_math_greater, tf_math_greater_equal, tf_logical_and, tf_math_logical_xor, tf_math_logical_or,\
    tf_argmin, tf_math_is_inf, tf_math_is_nan, tf_size, tf_not_equal, tf_where, tf_transpose, tf_gather_nd,\
    tf_multiply, tf_image_non_max_suppression, tf_ones_like, tf_stack, tf_concat

# Handle python 2.7 import error
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def convert_clip(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert clip layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.clip')
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for clip layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    clip_min = params.get('min')
    clip_max = params.get('max')
    if clip_min is None or clip_max is None:
        if len(node.input) == 1:
            raise UnsupportedLayer('Clip without max or min params')
        if len(node.input) > 1 and node.input[1] != '':
            clip_min = float(layers[node.input[1]])
        if len(node.input) == 3 and node.input[2] != '':
            clip_max = float(layers[node.input[2]])

    if clip_min is None and clip_max is None:
        raise UnsupportedLayer('Clip without max or min params')

    if clip_min is None:
        clip_min = tf.float32.min

    if clip_max is None:
        clip_max = tf.float32.max

    layers[node_name] = tf_clip_by_value(input_0, clip_min, clip_max, tf_name=f"{params['cleaned_name']}_clip")


def convert_log(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Log layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for log layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        import keras.backend as K
        return K.log(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_log")
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_neg(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Neg layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for log layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    layers[node_name] = tf_math_negative(input_0, tf_name=f"{params['cleaned_name']}_neg")


def convert_exp(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Exp layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for log layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        import keras.backend as K
        return K.exp(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_exp")
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_reduce_sum(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce sum.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce sum layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    if 'axes' not in params:
        axis = layers[node.input[1]]
    else:
        axis = params['axes']

    keep_dims = True
    if 'keepdims' in params:
        if params['keepdims'] == 0:
            keep_dims = False

    def target_layer(x, axis=axis, keep_dims=keep_dims):
        import keras.backend as K
        return K.sum(x, keepdims=keep_dims, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_reduce_sum")
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


def convert_reduce_mean(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce mean.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce mean layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    param_keepdims = params.get('keepdims', 1)
    keepdims = param_keepdims == 1
    axes = params['axes']
    layers[node_name] = K_mean(input_0, keepdims=keepdims, axis=axes, tf_name=f"{params['cleaned_name']}_mean")


def convert_reduce_max(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce max.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce max layer.')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, axis=params.get('axes'), keepdims=params['keepdims']):
        import keras.backend as K
        return K.max(x, keepdims=(keepdims == 1), axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_reduce_max")
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


def convert_reduce_min(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce max.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if params.get("axes") is not None:  # opset 13
        axes = params.get("axes")
    elif len(node.input) == 2:
        axes = layers.get(node.input[1])
    noop_with_empty_axes = bool(params.get("noop_with_empty_axes", False))
    keepdims = params.get("keepdims", True)
    if noop_with_empty_axes and params.get("axes") is None:
        layers[node_name] = layers[node.input[0]]
    else:
        layers[node_name] = tf_math_reduce_min(layers[node.input[0]], axis=axes, keepdims=keepdims,
                                               tf_name=f"{params['cleaned_name']}_min")


def convert_reduce_prod(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce max.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if params.get("axes") is not None:  # opset 13
        axes = params.get("axes")
    elif len(node.input) == 2:
        axes = layers.get(node.input[1])
    else:
        axes = None  # default is to reduce over all dimensions
    noop_with_empty_axes = bool(params.get("noop_with_empty_axes", False))
    keepdims = bool(params.get("keepdims", True))
    if noop_with_empty_axes and params.get("axes") is None:
        layers[node_name] = layers[node.input[0]]
    else:
        layers[node_name] = tf_math_reduce_prod(layers[node.input[0]],
                                                axis=axes,
                                                keepdims=keepdims,
                                                tf_name=f"{params['cleaned_name']}_reduce")


def convert_pow(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Pow layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 2:
        assert AttributeError('More than 2 inputs for pow layer.')
    layers[node_name] = tf_math_pow(layers[node.input[0]], layers[node.input[1]],
                                    tf_name=f"{params['cleaned_name']}_pow")


def convert_sqrt(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Sqrt layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for sqrt layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf_math_sqrt(input_0, tf_name=f"{params['cleaned_name']}_sqrt")


def convert_split(node, params, layers, lambda_func, node_name, keras_names):
    """
    Convert Split layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for split layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_names[0])
    axis = params.get("axis", 0)
    try:  # onnx opset12
        splits = params["split"]
    except KeyError as e:  # onnx opset 14
        if len(node.input) > 1:
            splits = layers[node.input[1]]
        else:
            if layers[node.input[0]].shape[axis] % 2 != 0:
                raise AttributeError("No splits supplied to the split block but there are uneven number of channels")
            else:
                splits = [layers[node.input[0]].shape[axis] // 2] * 2
    if not isinstance(splits, Iterable):
        # This might not work if `split` is a tensor.
        chunk_size = K.int_size(input_0)[axis] // splits
        splits = (chunk_size,) * splits
    cur = 0
    for i, split in enumerate(splits):
        if len(splits) > 1:
            node_name = params['_outputs'][i]

        def target_layer(x, axis=axis, start_i=cur, end_i=cur + split):
            slices = [slice(None, None)] * len(K.int_shape(x))
            slices[axis] = slice(start_i, end_i)
            return x[tuple(slices)]

        layers[node_name] = target_layer(input_0)
        cur += split


def convert_cast(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Cast layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.cast')

    if len(node.input) != 1:
        assert AttributeError('More than 1 input for cast layer.')

    if is_numpy(layers[node.input[0]]):
        logger.debug('Cast numpy array')

        cast_map = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            9: np.bool,
            10: np.float16,
            11: np.double,
        }
        cast_result = layers[node.input[0]]
        result = (layers[node.input[0]] == None)
        if isinstance(result, (bool, np.bool_)) and not result:
            cast_result = cast_map[params['to']](layers[node.input[0]])
        elif not isinstance(result, (bool, np.bool_)) and not np.any(result):
            cast_result = cast_map[params['to']](layers[node.input[0]])
        layers[node_name] = cast_result
    else:
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        check_cast_map = {
                1: tf.float32,
                2: tf.uint8,
                3: tf.int8,
                5: tf.int16,
                6: tf.int32,
                7: tf.int64,
                9: tf.bool,
                10: tf.float16,
                11: tf.double,
            }
        if input_0.dtype == check_cast_map[params['to']] and not isinstance(input_0, (tf.Tensor, np.ndarray)):
            # casting a tensor to the same dtype create placeholder:0 tensor which does not process well in engine
            # trying to ignore the conversion (since its identity) might result in wrong types due to the way
            # keras changes types on serialization and deserialization.
            # So we up-cast to the most informative type then downcast.
            # I'm Sorry.
            if input_0.dtype != tf.double:
                input_0 = tf_cast(input_0, tf.double, tf_name=f"{params['cleaned_name']}_precast")
            else:
                # We can add an If operation to the graph here if needed
                raise NotImplementedError("Does not support tf.double casting into itself")

        def target_layer(x, dtype=params['to'], k_name=f"{params['cleaned_name']}"):
            import tensorflow as tf
            cast_map = {
                1: tf.float32,
                2: tf.uint8,
                3: tf.int8,
                5: tf.int16,
                6: tf.int32,
                7: tf.int64,
                9: tf.bool,
                10: tf.float16,
                11: tf.double,
            }
            return tf_cast(x, cast_map[dtype], tf_name=f'{k_name}_cast')

        layers[node_name] = target_layer(input_0)


def convert_floor(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Floor layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for floor layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        # Floor is absent in keras.backend
        import tensorflow as tf
        return tf.floor(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_floor")
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_abs(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    res = tf_math_abs(input_0, tf_name=f'{params["cleaned_name"]}_abs')
    layers[node_name] = res


def convert_identity(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Identity layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for itentity layer.')

    layers[node_name] = layers[node.input[0]]


def convert_argmax(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ArgMax layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for argmax layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get("axis", -1)
    should_keep_dims = params.get("keepdims", True)

    argmax = tf_argmax(input_0, axis=axis, tf_name=f"{params['cleaned_name']}_argmax")
    if should_keep_dims:
        argmax = tf_expand_dims(argmax, axis=axis, tf_name=f"{params['cleaned_name']}_expand")
    layers[node_name] = argmax


def convert_argmin(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ArgMax layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for argmax layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get("axis", -1)
    should_keep_dims = params.get("keepdims", True)

    argmin = tf_argmin(input_0, axis=axis, tf_name=f"{params['cleaned_name']}_argmin")
    if should_keep_dims:
        argmin = tf_expand_dims(argmin, axis=axis, tf_name=f"{params['cleaned_name']}_argmin_unsqueeze")
    layers[node_name] = argmin


def convert_reduce_l2(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ReduceL2 layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce_l2 layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get("axes", [-1])
    keepdims = params.get("keepdims", 0)

    def target_layer(x, axis=axis, keepdims=keepdims):
        import tensorflow as tf
        if isinstance(axis, list) and len(axis) == 1:
            axis = axis[0]
        return tf.norm(x, axis=axis, keepdims=keepdims == 1)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_reduce_l2")
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_reciprocal(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf_math_reciprocal(input_0, tf_name=f"{params['cleaned_name']}_reciprocal")


def convert_not(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf_logical_not(input_0, tf_name=f"{params['cleaned_name']}_not")


def convert_less(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    if input_1.dtype == input_0.dtype and not isinstance(input_0, (tf.Tensor, np.ndarray)):
        if input_0.dtype != tf.double:
            # To see why this is needed, see inline comments on convert_cast
            input_0 = tf_cast(input_0, dtype=tf.double, tf_name=f"{params['cleaned_name']}_less_cast")
        else:
            raise NotImplementedError("Casting a tensor to itself is not supported")
    def target_layer(y, x=input_0):
        x = tf.cast(x, y.dtype)
        return tf.math.less(x, y)

    lambda_less = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_less")
    less_output = lambda_less(input_1)
    layers[node_name] = less_output


def convert_sign(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_sign(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_sign")


def convert_sin(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_sin(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_sin")


def convert_cosh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_cosh(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_cosh")


def convert_ceil(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_ceil(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_ceil")


def convert_acosh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_acosh(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_acosh")


def convert_acos(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_acos(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_acos")


def convert_asinh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_asinh(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_asinh")


def convert_asin(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_asin(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_asin")


def convert_atanh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_atanh(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_atanh")


def convert_tan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_tan(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_tan")


def convert_atan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_atan(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_atan")


def convert_sinh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_sinh(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_sinh")


def convert_less_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_less_equal(layers[node.input[0]], layers[node.input[1]],
                                           tf_name=f"{params['cleaned_name']}_less_equal")


def convert_bitwise_not(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_bitwise_invert(tf.cast(layers[node.input[0]], tf.int32),
                                          tf_name=f"{params['cleaned_name']}_bitwise_not")


def convert_bitwise_and(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_bitwise_bitwise_and(layers[node.input[0]], layers[node.input[1]],
                                               tf_name=f"{params['cleaned_name']}_bitwise_and")


def convert_bitwise_or(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_bitwise_bitwise_or(layers[node.input[0]], layers[node.input[1]],
                                              tf_name=f"{params['cleaned_name']}_bitwise_or")


def convert_bitwise_xor(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_bitwise_bitwise_xor(layers[node.input[0]], layers[node.input[1]],
                                               tf_name=f"{params['cleaned_name']}_bitwise_xor")


def convert_cosine(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf_cos(input_0, tf_name=f"{params['cleaned_name']}_cos")


def convert_greater(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_greater(layers[node.input[0]], layers[node.input[1]],
                                        tf_name=f"{params['cleaned_name']}_greater")


def convert_greater_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_greater_equal(layers[node.input[0]], layers[node.input[1]],
                                              tf_name=f"{params['cleaned_name']}_greater_equal")


def convert_and(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_logical_and(layers[node.input[0]], layers[node.input[1]],
                                       tf_name=f"{params['cleaned_name']}_and")


def convert_xor(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_logical_xor(layers[node.input[0]], layers[node.input[1]],
                                            tf_name=f"{params['cleaned_name']}_xor")


def convert_or(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_logical_or(layers[node.input[0]], layers[node.input[1]],
                                           tf_name=f"{params['cleaned_name']}_or")


def convert_trilu(node, params, layers, lambda_func, node_name, keras_name):
    input = layers[node.input[0]]
    k = 0
    if len(node.input) > 1:
        k = layers[node.input[1]]

    if "upper" in params and not params["upper"]:
        result = tf.experimental.numpy.tril(input, k)

    else:
        result = tf.experimental.numpy.triu(input, k)
    layers[node_name] = result


def convert_cumsum(node, params, layers, lambda_func, node_name, keras_name):
    exclusive = bool(params.get("exclusive", 0))
    reverse = bool(params.get("reverse", 0))
    layers[node_name] = tf.math.cumsum(layers[node.input[0]], layers[node.input[1]],
                                       exclusive=exclusive, reverse=reverse)


def convert_is_inf(node, params, layers, lambda_func, node_name, keras_name):
    if params.get("detect_negative") is not None or params.get("detect_negative") is not None:
        raise AttributeError("Unsupported params detected in isInf conversion: detect_negative/detect_positive")
    layers[node_name] = tf_math_is_inf(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_is_inf")


def convert_is_nan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_math_is_nan(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_is_nan")


def convert_size(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_size(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_size")


def convert_non_zero(node, params, layers, lambda_func, node_name, keras_name):
    input_tensor = layers[node.input[0]]
    condition = tf_not_equal(
        input_tensor,
        tf.zeros_like(input_tensor),
        tf_name=f"{params['cleaned_name']}_non_zero_unequal"
    )
    nonzero_indices = tf_where(condition, tf_name=f"{params['cleaned_name']}_non_zero_where")
    nonzero_result = tf_transpose(nonzero_indices, tf_name=f"{params['cleaned_name']}_non_zero_transpose")
    nonzero_result = tf_cast(nonzero_result, tf.int32, tf_name=f"{params['cleaned_name']}_non_zero_cast")
    layers[node_name] = nonzero_result
    # tf.experimental.numpy.nonzero(layers[node.input[0]]) was not giving the right results


def convert_gather_nd(node, params, layers, lambda_func, node_name, keras_name):
    input_tensor = layers[node.input[0]]
    indices_tensor = layers[node.input[1]]
    batch_dims = params.get("batch_dims", 0)
    # tesnsorflow implementation of gather_nd, in any case it fails please try also the pseudo_gathernd function here
    # instead. basically it flattens the params and use normal gather to simulate the result of gathernd
    res = tf_gather_nd(input_tensor, indices_tensor, batch_dims=batch_dims,
                       tf_name=f"{params['cleaned_name']}_gather_nd")
    layers[node_name] = res


def pseudo_gathernd(input_tensor, indices_tensor):
    params_shape = input_tensor.shape
    idx_shape = indices_tensor.shape
    idx_dims = idx_shape[-1]
    gather_shape = params_shape[idx_dims:]
    params_flat = tf.reshape(
        input_tensor,
        tf.concat([[-1], gather_shape], axis=0),
    )
    axis_step = tf.math.cumprod(
        params_shape[:idx_dims],
        exclusive=True,
        reverse=True,
    )

    NUMPY_DTYPES_TO_TF_DTYPES = {
        np.dtype('float16'): tf.float16,
        np.dtype('float32'): tf.float32,
        np.dtype('float64'): tf.float64,

        np.dtype('uint8'): tf.uint8,
        np.dtype('uint16'): tf.uint16,
        np.dtype('uint32'): tf.uint32,
        np.dtype('uint64'): tf.uint64,

        np.dtype('int8'): tf.int8,
        np.dtype('int16'): tf.int16,
        np.dtype('int32'): tf.int32,
        np.dtype('int64'): tf.int64,

        np.dtype('bool_'): tf.bool,
    }

    mul = tf.math.multiply(
        indices_tensor,
        tf.cast(
            axis_step,
            dtype=NUMPY_DTYPES_TO_TF_DTYPES[indices_tensor.dtype] \
                if isinstance(indices_tensor.dtype, np.dtype) else indices_tensor.dtype,
        ),
    )
    indices_flat = tf.reduce_sum(
        mul,
        axis=-1,
    )
    result_flat = tf.gather(
        params_flat,
        indices_flat,
    )
    if len(idx_shape) > 0 and len(idx_shape[:-1]) > 0 and idx_shape[:-1][0] is not None:
        pseudo_gathernd_res = tf.reshape(
            result_flat,
            tf.concat([idx_shape[:-1], gather_shape], axis=0),
        )
    else:
        pseudo_gathernd_res = result_flat

    return pseudo_gathernd_res


def convert_nms(node, params, layers, lambda_func, node_name, keras_name):
    scores = layers[node.input[1]]
    boxes = layers[node.input[0]]

    batch_size = boxes.shape[0]

    if batch_size is None:
        raise AttributeError("Onnx2kerras: NMS conversion does not support dynamic batch."
                             "Please change batch to static or remove NMS from model")
    center_point_box = params.get("center_point_box", 0)
    if center_point_box != 0:
        raise AttributeError("Onnx2kerras: We do not support the center_point_box parameter")

    iou_threshold = 0
    score_threshold = float('-inf')
    max_output_size = [2 ** 30]
    if len(node.input) > 2:
        max_output_size = [min(np.squeeze(layers.get(node.input[2], [2 ** 30])), 2 ** 30)]
    if len(node.input) > 3:
        iou_threshold = layers.get(node.input[3], [0])
    if len(node.input) > 4:
        score_threshold = ensure_float(layers.get(node.input[4], float('-inf')))
        if isinstance(score_threshold, np.ndarray):
            score_threshold = score_threshold[0]
    num_classes = scores.shape[1]
    all_results = []
    try:
        iou_threshold = iou_threshold[0]
    except IndexError: #iou threshold is already a scalar
        pass
    for batch in range(batch_size):
        for c_class in range(num_classes):
            indices = tf_image_non_max_suppression(boxes=boxes[batch],
                                                   scores=scores[batch, c_class],
                                                   max_output_size=tf.cast(max_output_size[0], tf.int32),
                                                   iou_threshold=iou_threshold,
                                                   score_threshold=score_threshold,
                                                   tf_name=f"{params['cleaned_name']}_nms_{batch}_{c_class}")
            ones_indices = tf_ones_like(indices, tf_name=f"{params['cleaned_name']}_nms_ones_{batch}_{c_class}")
            class_tensor = c_class * ones_indices
            batch_tensor = batch * ones_indices
            res = tf_stack([batch_tensor, class_tensor, indices], axis=-1
                           , tf_name=f"{params['cleaned_name']}_nms_stack_{batch}_{c_class}")
            all_results.append(res)
    layers[node_name] = tf_cast(tf_concat(all_results, axis=0, tf_name=f"{params['cleaned_name']}_nms_concat"),
                                dtype=tf.int64,
                                tf_name=f"{params['cleaned_name']}_nms_cast")


def convert_if(node, params, layers, lambda_func, node_name, keras_name):
    if len(layers[node.input[0]].shape) == 0:
        cond = layers[node.input[0]]
    else:
        cond = layers[node.input[0]][0]
    outputs = [layers[node.attribute[i].g.output[0].name] for i in range(2)]
    outputs_dtypes = [output.dtype for output in outputs]
    outputs_numpy_dtypes = [outputs_dtypes[i] if is_numpy(outputs[i]) else outputs_dtypes[i].as_numpy_dtype for i in
                            range(2)]
    if outputs_numpy_dtypes[0] != outputs_numpy_dtypes[1]:
        smallest_idx = np.argmin([np.iinfo(outputs_numpy_dtypes[i]).max for i in range(2)])
        if is_numpy(outputs[smallest_idx]):
            outputs[smallest_idx] = outputs[smallest_idx].astype(outputs_numpy_dtypes[1 - smallest_idx])
        else:
            outputs[smallest_idx] = tf_cast(outputs[smallest_idx], tf.as_dtype(outputs_dtypes[1 - smallest_idx]),
                                            tf_name=f"{params['cleaned_name']}_if_cast")
    in_vec = outputs[0]
    if is_numpy(in_vec):  # if this is a constant it would not be serialized well. connect it to input
        # f_then = lambda x: in_vec
        new_dtype = in_vec.dtype.type

        # The Tf conversion is required to pass args serialization in leap-model-parser
        def get_empty_array(x, dtype=new_dtype, keras_name=keras_name):
            return tf.convert_to_tensor(np.array([]), dtype=new_dtype, name=f'{params["cleaned_name"]}_if_empty_arr')

        if len(in_vec) == 0:  # empty arrays does not serialize well in lambdas.
            then_lambda = get_empty_array
        else:
            then_lambda = lambda x: in_vec
        lambda_layer = tf.keras.layers.Lambda(then_lambda, name=f"{params['cleaned_name']}_if_serizlize_arr_helper")
        if not K.is_keras_tensor(cond):
            raise NotImplementedError(
                "We do not support an if where both the then branch and the in-vector are constants")
        then_output = lambda_layer(cond)  # this assumes
    else:
        then_output = outputs[0]
    layers[node_name] = tf.keras.backend.switch(cond, then_output, outputs[1])


def convert_einsum(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    equation = params['equation'].decode('utf-8')

    if isinstance(input_0, tf.Tensor) and isinstance(input_1, tf.Tensor):
        layers[node_name] = tf.einsum(equation, *[input_0, input_1], name=keras_name)
    elif isinstance(input_0, tf.Tensor):
        layers[node_name] = OnnxEinsumLayer(equation, input_0, 0)(input_1, name=keras_name)
    elif isinstance(input_1 , tf.Tensor):
        layers[node_name] = OnnxEinsumLayer(equation, input_1, 1)(input_0, name=keras_name)
    else:
        layers[node_name] = OnnxEinsumLayer(equation, None, None)([input_0, input_1], name=keras_name)


