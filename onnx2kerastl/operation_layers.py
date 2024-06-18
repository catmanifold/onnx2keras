import logging

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

from .exceptions import UnsupportedLayer
from .utils import is_numpy, ensure_tf_type, ensure_float

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

    layers[node_name] = tf.clip_by_value(input_0, clip_min, clip_max)


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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
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

    layers[node_name] = tf.math.negative(input_0)


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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
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
    layers[node_name] = K.mean(input_0, keepdims=keepdims, axis=axes)


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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
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
        layers[node_name] = tf.math.reduce_min(layers[node.input[0]], axis=axes, keepdims=keepdims)


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
        layers[node_name] = tf.math.reduce_prod(layers[node.input[0]], axis=axes, keepdims=keepdims)


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
    layers[node_name] = tf.math.pow(layers[node.input[0]], layers[node.input[1]])


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
    layers[node_name] = tf.math.sqrt(input_0)


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

        def target_layer(x, dtype=params['to']):
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
            return tf.cast(x, cast_map[dtype])

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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_abs(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.math.abs(input_0)


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

    argmax = tf.argmax(input_0, axis=axis)
    if should_keep_dims:
        argmax = tf.expand_dims(argmax, axis=axis)
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

    argmin = tf.argmin(input_0, axis=axis)
    if should_keep_dims:
        argmin = tf.expand_dims(argmin, axis=axis)
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

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_reciprocal(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.math.reciprocal(input_0)


def convert_not(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.logical_not(input_0)


def convert_less(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    def target_layer(y, x=input_0):
        x = tf.cast(x, y.dtype)
        return tf.math.less(x, y)

    lambda_less = keras.layers.Lambda(target_layer, name=keras_name)
    less_output = lambda_less(input_1)
    layers[node_name] = less_output


def convert_sign(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.sign(layers[node.input[0]])


def convert_sin(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.sin(layers[node.input[0]])


def convert_cosh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.cosh(layers[node.input[0]])


def convert_ceil(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.ceil(layers[node.input[0]])


def convert_acosh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.acosh(layers[node.input[0]])


def convert_acos(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.acos(layers[node.input[0]])


def convert_asinh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.asinh(layers[node.input[0]])


def convert_asin(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.asin(layers[node.input[0]])


def convert_atanh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.asinh(layers[node.input[0]])


def convert_tan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.tan(layers[node.input[0]])


def convert_atan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.asin(layers[node.input[0]])


def convert_sinh(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.sinh(layers[node.input[0]])


def convert_less_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.less_equal(layers[node.input[0]], layers[node.input[1]])


def convert_bitwise_not(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.bitwise.invert(tf.cast(layers[node.input[0]], tf.int32))


def convert_bitwise_and(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.bitwise.bitwise_and(layers[node.input[0]], layers[node.input[1]])


def convert_bitwise_or(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.bitwise.bitwise_or(layers[node.input[0]], layers[node.input[1]])


def convert_bitwise_xor(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.bitwise.bitwise_xor(layers[node.input[0]], layers[node.input[1]])


def convert_cosine(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.cos(input_0)


def convert_greater(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.greater(layers[node.input[0]], layers[node.input[1]])


def convert_greater_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.greater_equal(layers[node.input[0]], layers[node.input[1]])


def convert_and(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.logical_and(layers[node.input[0]], layers[node.input[1]])


def convert_xor(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.logical_xor(layers[node.input[0]], layers[node.input[1]])


def convert_or(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.logical_or(layers[node.input[0]], layers[node.input[1]])


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
    layers[node_name] = tf.math.is_inf(layers[node.input[0]])


def convert_is_nan(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.math.is_nan(layers[node.input[0]])


def convert_size(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.size(layers[node.input[0]])


def convert_non_zero(node, params, layers, lambda_func, node_name, keras_name):
    input_tensor = layers[node.input[0]]
    condition = tf.not_equal(
        input_tensor,
        tf.zeros_like(input_tensor),
    )
    nonzero_indices = tf.where(condition)
    nonzero_result = tf.transpose(nonzero_indices)
    nonzero_result = tf.cast(nonzero_result, tf.int32)
    layers[node_name] = nonzero_result
    # tf.experimental.numpy.nonzero(layers[node.input[0]]) was not giving the right results


def convert_gather_nd(node, params, layers, lambda_func, node_name, keras_name):
    input_tensor = layers[node.input[0]]
    indices_tensor = layers[node.input[1]]
    batch_dims = params.get("batch_dims", 0)
    # tesnsorflow implementation of gather_nd, in any case it fails please try also the pseudo_gathernd function here
    # instead. basically it flattens the params and use normal gather to simulate the result of gathernd
    res = tf.gather_nd(input_tensor, indices_tensor, batch_dims=batch_dims)
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
            indices = tf.image.non_max_suppression(boxes=boxes[batch],
                                                   scores=scores[batch, c_class],
                                                   max_output_size=tf.cast(max_output_size[0], tf.int32),
                                                   iou_threshold=iou_threshold,
                                                   score_threshold=score_threshold)
            class_tensor = c_class * tf.ones_like(indices)
            batch_tensor = batch * tf.ones_like(indices)
            res = tf.stack([batch_tensor, class_tensor, indices], axis=-1)
            all_results.append(res)
    layers[node_name] = tf.cast(tf.concat(all_results, axis=0), dtype=tf.int64)


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
            outputs[smallest_idx] = tf.cast(outputs[smallest_idx], tf.as_dtype(outputs_dtypes[1 - smallest_idx]))
    in_vec = outputs[0]
    if is_numpy(in_vec):  # if this is a constant it would not be serialized well. connect it to input
        # f_then = lambda x: in_vec
        new_dtype = in_vec.dtype.type

        # The Tf conversion is required to pass args serialization in leap-model-parser
        def get_empty_array(x, dtype=new_dtype, keras_name=keras_name):
            return tf.convert_to_tensor(np.array([]), dtype=new_dtype, name=f'{keras_name}_output')

        if len(in_vec) == 0:  # empty arrays does not serialize well in lambdas.
            then_lambda = get_empty_array
        else:
            then_lambda = lambda x: in_vec
        lambda_layer = tf.keras.layers.Lambda(then_lambda, name=keras_name)
        if not K.is_keras_tensor(cond):
            raise NotImplementedError(
                "We do not support an if where both the then branch and the in-vector are constants")
        then_output = lambda_layer(cond)  # this assumes
    else:
        then_output = outputs[0]
    layers[node_name] = tf.keras.backend.switch(cond, then_output, outputs[1])


def convert_einsum(node, params, layers, lambda_func, node_name, keras_name):
    # input_0 = layers[node.input[0]]
    # input_1 = layers[node.input[1]]
    # equation = params['equation'].decode('utf-8')
    # layers[node_name] = tf.einsum(equation, *[input_0, input_1], name=keras_name)
    input_0 = tf.expand_dims(layers[node.input[0]], axis=2)
    input_1 = tf.expand_dims(layers[node.input[1]], axis=0)
    layers[node_name] = tf.multiply(input_0, input_1, name=keras_name)
