from keras.layers import Layer, TFOpLambda
import tensorflow as tf
import numpy as np


# this custom layer needed because of a tensorflow bug on einsum serielization
class OnnxEinsumLayer(Layer):
    """
    Layer wrapping a single tf.einsum operation.

    Usage:
    x = EinsumLayer("bmhwf,bmoh->bmowf")((x1, x2))
    """

    def __init__(self, equation: str, constant_input, constant_place):
        super().__init__()
        self.equation = equation
        if constant_input is not None:
            if hasattr(constant_input, 'numpy'):
                constant_input = constant_input.numpy()
            if not isinstance(constant_input, np.ndarray):
                constant_input = np.array(constant_input)
            self.constant_input = constant_input
        self.constant_place = constant_place

    def call(self, inputs, *args, **kwargs):
        if self.constant_input is not None:
            if self.constant_place == 1:
                inputs = [inputs, self.constant_input]
            else:
                inputs = [self.constant_input, inputs]

        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        return {
            "equation": self.equation,
            "constant_input": self.constant_input,
            "constant_place": self.constant_place
        }
