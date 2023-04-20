#!/usr/bin/python3
"""
Sequence to sequence time series modeling.

Authors: Alexander Katrompas
Organization: Texas State University

Usage: main.py [-vg]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)
g: graphing mode (optional)

For temporal attention signature research and analysis
 - STEP=1, SEC2SEQ=False, CAPATTN=1
 - SHUFFLE = True or False depending on type of data (usually true for sig generation)
 - special case, ECG data, SLIDE=True, otherwise False
 - run this file
 - run sig-process.py in scripts/

For Seq2Seq research and analysis
 -
 - run this file first with ... to get A/P seq2vec for analysis and comparison to seq2seq
     - SEC2SEQ=False
     - STEP=0 or STEP=1, whichever is best
     - SHUFFLE= True or False, whichever is best
 - then run this file with ... to get A/P seq2seq for analysis and comparison
     - SEC2SEQ=True
     - STEP=0
     - CAPATTN=1 (if you want to run attention voting), else 0
 - for seq2vec to seq2seq analysis
    - run hard vote
    - run soft vote without attention
    - run soft vote with attention (if you don't need this, run with CAPATTN=0 for speed)
    - run stacking
    - compare all five
"""

###########################
# imports
###########################

# python libs
import os
import shutil
from pathlib import Path
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
import numpy as np


# local libs
import cfg
import functions as fn
from attnSelf import AttnSelf
from attention import Attention
from cbTrain import CbTrain
from cbPredict import CbPredict

###########################
# set-up
###########################
np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True)
fn.delete_attn_capure_files()
fn.delete_ap_files()

# get command line arguments
verbose, graph = fn.get_args()
fn.print_setup(verbose, graph) if verbose else print("running silent...")

###########################
# get data
###########################

# load files into train, test
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = fn.load_data(cfg.TRAIN, cfg.VALID, cfg.TEST,
                                                                       norm=False, outcols=cfg.OUTPUT,
                                                                       index=cfg.INDEX, header=cfg.HEADER)
train_timesteps = train_X.shape[0]
valid_timesteps = valid_X.shape[0]
test_timesteps = test_X.shape[0]

if verbose:
    print("files loaded...")
    print("  Train:", Path(cfg.TRAIN).name)
    print("  Valid:", Path(cfg.VALID).name)
    print("  Test: ", end = "")
    print(Path(cfg.TEST).name) if cfg.TEST else print(" validation set")
    print()
    print("Shapes loaded...")
    print("  train X:", train_X.shape)
    print("  train Y:", train_Y.shape)
    print("  valid_X:", valid_X.shape)
    print("  valid_Y:", valid_Y.shape)
    print("  test_X:", test_X.shape)
    print("  test_Y:", test_Y.shape)
    print("  Sequence length:", cfg.SEQLENGTH)
    print("  Total testing time steps:", test_timesteps)
    print()

if cfg.SLIDE:
    if verbose: print("  shaping train... ", end="")
    train_X, train_Y, randnums_train = fn.shape_slide_3D_data(train_X, train_Y, cfg.SEQLENGTH, cfg.STEP, cfg.SEC2SEQ)
    if verbose: print("shaping valid... ", end="")
    valid_X, valid_Y, randnums_valid = fn.shape_slide_3D_data(valid_X, valid_Y, cfg.SEQLENGTH, cfg.STEP, cfg.SEC2SEQ)
    if verbose: print("shaping test... ", end="")
    test_X, test_Y, randnums_test = fn.shape_slide_3D_data(test_X,  test_Y, cfg.SEQLENGTH, 1, cfg.SEC2SEQ)
    if verbose: print("done shaping.\n")

    if cfg.RSAVE:
        np.savetxt(cfg.TRAIN_RNS, randnums_train, delimiter=',', fmt='%d')
        np.savetxt(cfg.VALID_RNS, randnums_valid, delimiter=',', fmt='%d')
        np.savetxt(cfg.TEST_RNS, randnums_test, delimiter=',', fmt='%d')
        fn.save3d_to_csv(train_X, cfg.TRAIN_X_RESHAPED,'%f')
        fn.save3d_to_csv(train_Y, cfg.TRAIN_Y_RESHAPED,'%f')
        fn.save3d_to_csv(valid_X, cfg.VALID_X_RESHAPED,'%f')
        fn.save3d_to_csv(valid_Y, cfg.VALID_Y_RESHAPED,'%f')
        fn.save3d_to_csv(test_X, cfg.TEST_X_RESHAPED,'%f')
        fn.save3d_to_csv(test_Y, cfg.TEST_Y_RESHAPED,'%f')
else:
    # this is the special case for single channel fixed seqlen (e.g. ECG data)
    train_X = train_X.reshape(train_X.shape[0],cfg.SEQLENGTH,1)
    valid_X = valid_X.reshape(valid_X.shape[0],cfg.SEQLENGTH,1)
    test_X = test_X.reshape(test_X.shape[0],cfg.SEQLENGTH,1)
    # Y doesn't need to be reshaped

# must do these after reshape because some instances might fall off the back end during shaping
test_timesteps = test_X.shape[0]
if cfg.SEC2SEQ:
    total_predictions = test_X.shape[0] * test_X.shape[1]
else:
    total_predictions = test_X.shape[0]

if verbose:
    print("reshaped...")
    print("  train X", train_X.shape, train_X.shape[0]*train_X.shape[1], ":", round(train_X.shape[0]*train_X.shape[1] / train_timesteps,2))
    print("  train Y", train_Y.shape, train_X.shape[0]*train_X.shape[1])
    print("  valid_X", valid_X.shape, valid_X.shape[0]*valid_X.shape[1], ":", round(valid_X.shape[0]*valid_X.shape[1] / valid_timesteps,2))
    print("  valid_Y", valid_Y.shape, valid_X.shape[0]*valid_X.shape[1])
    print("  test_X", test_X.shape, test_X.shape[0]*test_X.shape[1], ":", round(test_X.shape[0]*test_X.shape[1] / test_timesteps,2))
    if cfg.SEC2SEQ:
        print("  test_Y", test_Y.shape, test_Y.shape[0] * test_Y.shape[1])
    else:
        print("  test_Y", test_Y.shape)
    print("  Sequence length:", cfg.SEQLENGTH)
    print("  Total testing time steps:", test_timesteps)
    print("  Total testing predictions:", total_predictions)
    print()

#############################
# Build Model
#############################
inp = layers.Input(shape=(cfg.SEQLENGTH, train_X.shape[2]))

if cfg.BIDIRECTIONAL:
    x = layers.Bidirectional(layers.LSTM(cfg.LSTM, return_sequences=True, dropout=cfg.DROPOUT))(inp)
else:
    x = layers.LSTM(cfg.LSTM, return_sequences=True, dropout=cfg.DROPOUT)(inp)

attn = Attention(return_sequences=True)(x)
attnSelf = AttnSelf(cfg.SEQLENGTH)(x)
out = layers.Concatenate(axis=-1)([attnSelf, attn])
del attn
del attnSelf

if not cfg.SEC2SEQ:
    out = layers.Flatten()(out)

out = layers.Dense(cfg.DENSE1, activation='relu')(out)
out = layers.Dropout(cfg.DROPOUT)(out)
out = layers.Dense(cfg.DENSE2, activation='relu')(out)
out = layers.Dropout(cfg.DROPOUT)(out)

out = layers.Dense(cfg.OUTPUT, activation='softmax')(out)
model = tf.keras.Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=cfg.EAGERLY)

if verbose:
    print("Epochs:", cfg.EPOCHS)
    model.summary()

#############################
# Train Model
#############################
cbTrain = CbTrain()
early_stopping = EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True, verbose=verbose)
history = model.fit(train_X, train_Y,
            #batch_size = cfg.BATCH_SIZE,
            epochs=cfg.EPOCHS,
            verbose=verbose,
            shuffle=cfg.SHUFFLE,
            validation_data=(valid_X, valid_Y),
            callbacks=[early_stopping,cbTrain])

if cfg.SAVEMODEL:
    if verbose: print("Model saving mode...", end="")
    model.save(cfg.SAVEMODELPATH)
    if verbose: print("Model saved.\n")
elif os.path.exists(cfg.SAVEMODELPATH):
    shutil.rmtree(cfg.SAVEMODELPATH)

#############################
# Graphing Loss
#############################
if graph and (len(history.history['loss']) and len(history.history['val_loss'])):
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

###################################
# Prediction and Validation
###################################

if verbose:
    if cfg.PREDICT:
        print("Beginning validation prediction.")
        if cfg.EAGERLY:
            print("Capturing attention enabled.")
        else:
            print("Capturing attention disabled.")
            print("To capture attention, network must be set to run eagerly in cfg.py.")
    else:
        print("No predictions generated. Check cfg.py if you expected predictions.")

if cfg.PREDICT:
    if cfg.EAGERLY: fn.create_attn_capure_files()

    if verbose:
        print()
        print("Total sequences to predict:", test_timesteps)
        print("Total outputs to predict:", total_predictions)

    #######################################################################
    # predictions process outputs a 3D array in both binary and multi-label
    #######################################################################
    predictionsraw = []

    if verbose:
        tenpercent = int(test_timesteps * .1)
        percent = -10

    for i in range(test_timesteps):
        sequence = test_X[i]
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        if cfg.EAGERLY: # capture attention
            if cfg.CAPATTN == 1: # capture general attention
                cbPredict = CbPredict()
            elif cfg.CAPATTN == 2:  # capture self attention
                cbPredict = CbPredict(sa=True)
            yhat = model.predict(sequence, callbacks=[cbPredict])  # 3D in seq2seq, 2D in seq2vec
        else: # do not capture attention
            yhat = model.predict(sequence)  # 3D in seq2seq, 2D in seq2vec

        if yhat.ndim == 3:
            yhat = yhat.reshape(yhat.shape[1],yhat.shape[2]) # make 2D...

        predictionsraw.append(yhat) # to put back into the final 3D structure

        if verbose and not (float(i) % tenpercent):
            percent += 10
            print(percent, "%...", sep="")

    predictionsraw = np.array(predictionsraw)
    if verbose:
        print("Done.", predictionsraw.shape[0], "predictions made.")
        print()

    ###############################
    # convert predictions to binary
    ###############################
    if cfg.VERBOSE: print("Post-processing predictions...", end="")

    # since 3D array, first turn into 2D array, rows are time-steps, columns are output vector
    predictionsraw = predictionsraw.reshape(predictionsraw.shape[0] * predictionsraw.shape[1], predictionsraw.shape[2])
    predictionscat = [] # to hold the categorical predictions

    # then turn into 1D array of argmax values for categorical output
    for value in predictionsraw:
        cat = np.argmax(value)
        predictionscat.append(cat) # gives back the categorical number from the one hot output
    predictionscat = np.array(predictionscat)

    # turn test_Y into categorical output for comparision to predicted categorical
    if test_Y.ndim == 3:
        temp_test_Y = test_Y.reshape(test_Y.shape[0] * test_Y.shape[1], test_Y.shape[2])
    else:
        temp_test_Y = np.copy(test_Y)
    test_Y = []
    for value in temp_test_Y:
        test_Y.append(np.argmax(value))
    del temp_test_Y
    test_Y = np.array(test_Y)

    if verbose:
        print("complete.\n")
        print("Writing predictions to files...")

    ##################
    # save predictions
    ##################

    if cfg.SEC2SEQ:
        # save raw output, one complete set of cfg.OUTPUT per line
        # only needed in seq2seq for soft and attention voting
        # not written for seq2vec model
        fout = open(cfg.PREDICTRAW, "w")
        for i in range(total_predictions):
            line = ""
            for j in range(cfg.OUTPUT):
                line += str(predictionsraw[i][j]) + ","
            fout.write(line[:-1] + "\n")
        fout.close()

    # save categorical processed output, single actual versus predicted per line
    # different file names so we don't confuse seq2seq and seq2vec
    if cfg.SEC2SEQ:
        fout = open(cfg.ACTUALPREDICTEDS, "w")
    else:
        fout = open(cfg.ACTUALPREDICTEDV, "w")
    for i in range(total_predictions):
        fout.write(str(test_Y[i]) + "," + str(predictionscat[i]) + "\n")
    fout.close()

    if cfg.SEC2SEQ:
        # save act/pred, one whole sequence per line, all actual then all predicted
        # don't need this if se2vec since cfg.ACTUALPREDICTEDS is sinlge output per sequence
        fout = open(cfg.ACTUALPREDICTEDSEQ, "w")
        for i in range(test_timesteps):
            line = ""
            for j in range(cfg.SEQLENGTH):
                line += str(test_Y[i*cfg.SEQLENGTH+j]) + ","
            for j in range(cfg.SEQLENGTH):
                line += str(predictionscat[i*cfg.SEQLENGTH+j]) + ","
            fout.write(line[:-1] + "\n")
        fout.close()

    if verbose:
        print("complete.\n")

    ############################################
    # analysis of predictions
    ############################################
    if verbose:
        print("Analysis of test data predictions...")
        print('  Sequence length: %d' % cfg.SEQLENGTH)
        print('  Test strides: 1')
        print("A/P saved:", end="")
        if cfg.SEC2SEQ:
            print(os.path.basename(cfg.ACTUALPREDICTEDS), ",", os.path.basename(cfg.ACTUALPREDICTEDSEQ))
            fn.print_stats(cfg.ACTUALPREDICTEDS, cfg.OUTPUT)
        else:
            print(os.path.basename(cfg.ACTUALPREDICTEDV))
            fn.print_stats(cfg.ACTUALPREDICTEDV, cfg.OUTPUT)
