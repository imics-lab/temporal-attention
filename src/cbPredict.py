"""
Sequence to sequence time series modeling.
Prediction callback for capturing and saving attention.

Authors: Alexander Katrompas
Organization: Texas State University

"""

import tensorflow as tf
import os
import numpy as np
import cfg

class CbPredict(tf.keras.callbacks.Callback):
    def __init__(self, sa=False, verbose = False):
        super().__init__()
        self.__verbose = verbose
        self.__sa = sa

        # files are created empty in main first
        # by calling create_attn_capture_files()
        # bad design, but it works for now

        if os.path.exists(cfg.TATTN):
            self.fout1 = open(cfg.TATTN, "a")
        else:
            raise Exception("file not found:", cfg.TATTN)

        if not self.__sa:
            if os.path.exists(cfg.TATTN_NORM):
                self.fout2 = open(cfg.TATTN_NORM, "a")
            else:
                raise Exception("file not found:", cfg.TATTN_NORM)

    def on_predict_begin(self, logs=None):
        pass

    # will accept and write one seq at a time
    # should only execute at the end of sequence
    def on_predict_end(self, logs=None):
        if not self.__sa:
            # capture general attention
            em = self.model.get_layer("attention").em.numpy()
            em = em.flatten()

            for value in em:
                self.fout1.write(str(value) + "\n")

            if np.min(em) == np.max(em):
                # accounting for no min and max being equal which will fault
                seq_norm = np.full([cfg.SEQLENGTH], np.max(em))
            else:
                seq_norm = (em - np.min(em)) / (np.max(em) - np.min(em))

            for value in seq_norm:
                self.fout2.write(str(value)+"\n")
        else:
            # capture self attention
            ems = self.model.get_layer("attn_self").ems.numpy()
            ems = ems.reshape(ems.shape[1],ems.shape[2]) # will be seq x seq

            for row in ems:
                line = ""
                for item in row:
                    line += str(item) + ","
                self.fout1.write(line[:-1] + "\n")

        self.fout1.close()
        if not self.__sa:
            self.fout2.close()
