from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import ops

def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    num_calls=1
    try:
        reverse_gradient.num_calls =reverse_gradient.num_calls+ 1
    except AttributeError:
        reverse_gradient.num_calls = num_calls
        num_calls=num_calls+1

    grad_name = "GradientReversal_%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op,grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))