"""
Run Time Configuration

Authors: Alexander Katrompas
Organization: Texas State University
"""

from os import getcwd

# ###############################
# data configuration
# ###############################
DATAPATH = getcwd() + '/../data/'
RESULTSPATH = getcwd() + '/../results/'
SAVEMODELPATH = RESULTSPATH + 'model/'
SAVESIGSPATH = RESULTSPATH + 'sigs/'

#set this per data file
DATA = 3

if DATA == 1:
    TRAIN = DATAPATH + "smartfall_train_1h.csv"
    VALID = DATAPATH + "smartfall_test_1h.csv"
    TEST = ""
    OUTPUT = 2
    INDEX = None
    HEADER = 0
    SEQLENGTH = 40
    LSTM = 128
elif DATA == 1:
    TRAIN = DATAPATH + "aq-train_1h.csv"
    VALID = DATAPATH + "aq-valid_1h.csv"
    TEST = ""
    OUTPUT = 2
    INDEX = 0
    HEADER = 0
    SEQLENGTH = 8
    LSTM = 256
elif DATA == 3:
    TRAIN = DATAPATH + "lakeoxygen_1ft_train_1h.csv"
    VALID = DATAPATH + "lakeoxygen_1ft_valid_1h.csv"
    TEST =  "" # optional, if none given valid set will be used for test
    OUTPUT = 3
    INDEX = None
    HEADER = None
    SEQLENGTH = 6
    LSTM = 256
elif DATA == 4:
    TRAIN = DATAPATH + "weatherAUStrain_1h.csv"
    VALID = DATAPATH + "weatherAUSvalid_1h.csv"
    TEST = ""  # optional, if none given valid set will be used for test
    OUTPUT = 2
    INDEX = 0
    HEADER = 0
    SEQLENGTH = 5
    LSTM = 256
elif DATA == 5:
    TRAIN = DATAPATH + "heart_train_1h.csv"
    VALID = DATAPATH + "heart_valid_1h.csv"
    TEST = DATAPATH + "heart_test_1h.csv"
    OUTPUT = 5
    INDEX = None
    HEADER = None
    SEQLENGTH = 187
    LSTM = 256
else:
    raise ValueError("data file error")
# ###############################
# command line parameter defaults
# ###############################
VERBOSE = False
GRAPH = False

# ###############################
# Most commonly changed
# ###############################
EPOCHS = 600
PATIENCE = 110
SEC2SEQ = False
CAPATTN = 0 # capture attention: 0 = no, 1 = general, 2 = self

# ###############################
# Hyperparameters
# ###############################

# STEP = 0 --> random slide, use for train and test, valid is hardcoded to always be 1
# STEP = 1 --> overlap of seqlen-1 (like time generator), valid is hardcoded to always be 1
# currently not using an step other than 0 and 1
STEP = 0 if SEC2SEQ else 1 # do not set this manually, use SEC2SEQ

SHUFFLE = False # should always be false in seq2seq work, could be either in seq2vec or sig work
BIDIRECTIONAL = True
DROPOUT = .25
# the following set automatically based on most common best performance
if BIDIRECTIONAL: DENSE1 = LSTM * 2
else: DENSE1 = LSTM
DENSE2 = DENSE1 / 2
EAGERLY = True if CAPATTN else False # do not set this manually, use CAPATTN

# ###############################
# Data
# ###############################

# slide means there will be overlap of size step (the common case)
SLIDE = True # currently this should only be set to False when running fixed sequence data like ECG
RSAVE = False # save the shaped data for debug
BINTHRESHOLD = 0.5 # determines how soft output is converted to one hot
PREDICT = True
SAVEMODEL = True # only need this for seq2seq sa voting

# ###############################
# Files
# ###############################

# save raw output from TEST, one complete set of cfg.OUTPUT per line
# needed in seq2seq for soft, attention voting, stacking
# not written for seq2vec modeling prediction
PREDICTRAW = RESULTSPATH + "predicted_raw.csv"

# save raw output from validation file, one complete sequence per line
# written and used by stack.py
PREDICTRAWVALID = RESULTSPATH + "predicted_raw_valid.csv"

# actual versus predicted, only one of these two is written during training and prediction
# ACTUALPREDICTEDV  is not deleted so it can be kept for analysis against seq2seq ACTUALPREDICTEDS
# if seq2vec, each line is a single point actual and prediction
ACTUALPREDICTEDV = RESULTSPATH + "actual_predicted_v.csv"
# if seq2seq, entire sequence is written, one actual and prediction value per line
ACTUALPREDICTEDS = RESULTSPATH + "actual_predicted_s.csv"

# ACTUALPREDICTED converted to one sequence of actual and one sequence of prediction per line
# used by hard vote as input
ACTUALPREDICTEDSEQ = RESULTSPATH + "actual_predicted_seq.csv"

# output of seq2seq conversions using hard-vote, soft-vote, attn-vote, etc.
ACTUALPREDICTEDHARDVEC = RESULTSPATH + "actual_predicted_hard_vec.csv"
ACTUALPREDICTEDSOFTVEC = RESULTSPATH + "actual_predicted_soft_vec.csv"
ACTUALPREDICTEDATTNVEC = RESULTSPATH + "actual_predicted_attn_vec.csv"

TATTN = RESULTSPATH + "tattn.csv" # all general attention layer activations
TATTN_NORM = RESULTSPATH + "tattn_norm.csv" # normalization of vattend.csv on a per seq basis
TATTN_NORM_PROCESSED = RESULTSPATH + "tattn_norm_processed.csv" # normalized sequences writen out by sequence
TATTN_SIGNATURES = RESULTSPATH + "tattn_signatures.csv"

if RSAVE:
    TRAIN_RNS = RESULTSPATH + "trains_rns.csv"
    VALID_RNS = RESULTSPATH + "valid_rns.csv"
    TEST_RNS = RESULTSPATH + "test_rns.csv"
    TRAIN_X_RESHAPED = RESULTSPATH + "train_X_reshaped.csv"
    TRAIN_Y_RESHAPED = RESULTSPATH + "train_Y_reshaped.csv"
    VALID_X_RESHAPED = RESULTSPATH + "valid_X_reshaped.csv"
    VALID_Y_RESHAPED = RESULTSPATH + "valid_Y_reshaped.csv"
    TEST_X_RESHAPED = RESULTSPATH + "test_X_reshaped.csv"
    TEST_Y_RESHAPED = RESULTSPATH + "test_Y_reshaped.csv"
