"""
Regular Simple Attention Layer

Authors: Alexander Katrompas
Organization: Texas State University

"""

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.em = None
        self.__return_sequences = return_sequences
        self.__W = None
        self.__b = None
        self.__out = None
        super(Attention,self).__init__()

    def build(self, input_shape):
        self.__W =self.add_weight(name="att_weight", shape=(input_shape[-1],1))
        self.__b =self.add_weight(name="att_bias", shape=(input_shape[1],1))
        super(Attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.__W)+self.__b)
        attention_weights = K.softmax(e, axis=1)

        self.em = attention_weights + 0 # for capture

        self.__out = x * attention_weights
        if not self.__return_sequences:
            self.__out = K.sum(self.__out, axis=1)
        return self.__out
