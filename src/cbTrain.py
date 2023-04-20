"""
Sequence to sequence time series modeling.
Training callback, currently used only to print start and stop.

Authors: Alexander Katrompas
Organization: Texas State University

"""
import tensorflow as tf

class CbTrain(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_begin(self, logs=None):
        print()
        print("****************")
        print("Training Started")

    def on_train_end(self, logs=None):
        print("Training Complete")
        print("*****************")
        print()
