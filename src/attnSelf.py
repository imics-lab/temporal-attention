"""
Self-Attension Layer (used in AIME paper)

Authors: Alexander Katrompas
Organization: Texas State University

Stand alone self-attendion layer class for use with LSTM layer
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten, Activation, Permute
from tensorflow.keras.layers import Permute
from tensorflow.keras import backend as K

class AttnSelf(Layer):
    """
    Stand alone self-attendion layer class for use with LSTM layer, conforming
    to the transformer concept from Attention Is All You Need (Vaswani 2017)
    https://arxiv.org/abs/1706.03762

    @param (int) alength: attention length
    @param (bool) return_sequences: return sequences true (default) or false
    """
    def __init__(self, alength, return_sequences = True):
        self.__alength = alength
        self.__return_sequences = return_sequences
        self.ems = None
        self.__W1 = None
        self.__W2 = None
        super(AttnSelf, self).__init__()

    def build(self, input_shape):
        self.__W1 = self.add_weight(name='W1',
                                  shape=(self.__alength, input_shape[2]),
                                  initializer='random_uniform',
                                  trainable=True)
        self.__W2 = self.add_weight(name='W2',
                                  shape=(input_shape[1], self.__alength),
                                  initializer='random_uniform',
                                  trainable=True)
        super(AttnSelf, self).build(input_shape)

    def call(self, inputs):
        W1, W2 = self.__W1[None, :, :], self.__W2[None, :, :]

        hidden_states_transposed = Permute(dims=(2, 1))(inputs)
        e = tf.matmul(W1, hidden_states_transposed)
        e = Activation('tanh')(e)

        attention_weights = tf.matmul(W2, e)
        attention_weights = Activation('softmax')(attention_weights)

        self.ems = attention_weights + 0  # for capture

        embedding_matrix = tf.matmul(attention_weights, inputs) # will be size seq x lstm
        if not self.__return_sequences:
            embedding_matrix = K.sum(embedding_matrix, axis=1)
        return embedding_matrix

# produces seq by seq matrix relating each
# time-step's to each time-step's importance
#
# For example for a seq length 6
# 0-to-0, 0-to-1, 0-to-2, 0-to-3, 0-to-4, 0-to-5
# 1-to-0, 1-to-1, 1-to-2, 1-to-3, 1-to-4, 1-to-5
# 2-to-0, 2-to-1, 2-to-2, 2-to-3, 2-to-4, 2-to-5
# 3-to-0, 3-to-1, 3-to-2, 3-to-3, 3-to-4, 3-to-5
# 4-to-0, 4-to-1, 4-to-2, 4-to-3, 4-to-4, 4-to-5
# 5-to-0, 5-to-1, 5-to-2, 5-to-3, 5-to-4, 5-to-5
