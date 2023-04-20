#!/usr/bin/python3
"""
Process general attention into signatures

Authors: Alexander Katrompas
Organization: Texas State University

Usage: sig-process.py [-v]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)

Input
 - TATTN_NORM
 - ACTUALPREDICTEDV

Output File: tattn_norm_processed.csv (used to be vattend_sum_norm_proc.csv)
 - seq length of attn scores
 - importance vector
 - output vector over sequence
 - which is sent to attn_analysis.py
"""

import sys
import os
import silence_tensorflow.auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 5,2
sys.path.insert(1, os.getcwd() + '/../src/')

import cfg
import functions as fn
import scrfunctions as sfn

# #####################################################
# Set up for processing
# #####################################################
np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True)
argc = len(sys.argv)
if argc >= 2 and sys.argv[1] == '-v':
    VERBOSE = True
else:
    VERBOSE = False

if not os.path.exists(cfg.TATTN_NORM):
    raise FileNotFoundError(os.path.basename(cfg.TATTN_NORM) + " not found.")
if not os.path.exists(cfg.ACTUALPREDICTEDV):
    raise FileNotFoundError(os.path.basename(cfg.ACTUALPREDICTEDV) + " not found.")

if VERBOSE:
    print(f"Processing files: {os.path.basename(cfg.TATTN_NORM)}, {os.path.basename(cfg.ACTUALPREDICTEDV)}.")
    print("Writing", os.path.basename(cfg.TATTN_NORM_PROCESSED)) # file to be produced for analysis
    print("Outputs:", cfg.OUTPUT)
    print("Sequence length:", cfg.SEQLENGTH)

attnlines = fn.getFileLines(cfg.TATTN_NORM)
aplines = fn.getFileLines(cfg.ACTUALPREDICTEDV)

# sanity check
if attnlines / cfg.SEQLENGTH != aplines and not (attnlines % cfg.SEQLENGTH):
    raise ValueError("Mismatch in number of lines between {} and {}.", os.path.basename(cfg.TATTN_NORM), os.path.basename(cfg.ACTUALPREDICTEDV))

# #####################################################
# Processing attention into sequenced and ranks
# #####################################################

# file to be processed
fin = open(cfg.TATTN_NORM, "r")

# file to be produced for analysis
fout = open(cfg.TATTN_NORM_PROCESSED, "w")

# get actual and predicted
aps = pd.read_csv(cfg.ACTUALPREDICTEDV, header=None).to_numpy()

# create arrays to be used later in analysis for confidence
attnscores = np.full([aplines, cfg.SEQLENGTH], -1, dtype=float)
ranks = np.full([aplines, cfg.SEQLENGTH], -1, dtype=float)

# write out header for the output file
for i in range(cfg.SEQLENGTH):
    fout.write("attn " + str(i) + ",")
for i in range(cfg.SEQLENGTH):
    fout.write("rank " + str(i) + ",")
fout.write("act,pred\n")

for i, ap in enumerate(aps):

    # write out to file normalized attention scores, 1 sequence per line
    line = ""
    for j in range(cfg.SEQLENGTH):
        line += fin.readline()[:-1] + ","
    fout.write(line)

    # save numeric attention scores in numpy array for use in analysis
    attnscores[i][:] = fn.text_to_numpy(line[:-1].split(','))

    # use that line to calculate importance vector and write it
    sranking = sfn.get_ranks(attnscores[i])
    # save numeric ranks in numpy array for use in analysis
    ranks[i][:] = sranking
    # write out to file ranks, 1 sequence per line
    fout.write(fn.numpy_to_text(sranking) + ",")

    # write out actual and predicted
    fout.write(str(ap[0]) + "," + str(ap[1])+ "\n")

fin.close()
fout.close()

if VERBOSE:
    print(f"Done.\nAttention values, ranks, actual, predicted written to {os.path.basename(cfg.TATTN_NORM_PROCESSED)}.\n")
    print("Beginning confidence analysis...")

# ############################################################
# Sorting attention scores and rankings by label and
# getting confidence measure for validation of the signatures.
# ############################################################

# get all labels and total of each label
labels, labelcounts = np.unique(aps[:, 1],return_counts=True)
print("label counts", labelcounts)

# sanity check
total_labels = sum(labelcounts)
if total_labels != aplines or len(labels) != cfg.OUTPUT:
    raise ValueError("Label count mismatch.")

# make array to hold ranks per label
ranklists = []
attentionlists = []
for i in range(cfg.OUTPUT):
    ranklists.append([])
    attentionlists.append([])

# sort the importance vectors and attention scores by label
for i in range(total_labels):
    ranklists[aps[i][1]].append(ranks[i])
    attentionlists[aps[i][1]].append(attnscores[i])

# the following code produces matrices, per label
# for example, for seqlen = 6, for each label
# 0 [  0  27  38  21  35   2]
# 1 [  0   0   4  77  38   4]
# 2 [  0  87  33   3   0   0]
# 3 [123   0   0   0   0   0]
# 4 [  0   9  48  21  45   0]
# 5 [  0   0   0   1   3 119]]

# read as such, for example, for this label...
# time-step 3 is importance 0 (least important) 123 times (100% - high confidence)
# time-step 5 is importance 5 (most important) 119 times (96.7% - high confidence)
# etc
# the only purpose to the rank matrix is to generate confidence scores

rankmatrices = []
for list in ranklists:
    rankmatrix = np.array(list)
    for i in range(cfg.SEQLENGTH):
        occurrences = np.count_nonzero(rankmatrix == i, axis=0)
        rankmatrices.append(occurrences)
# at this point, rankmatrices is not shaped by label and is perpendicular, needs transposed
rankmatrices = np.array(rankmatrices).reshape([cfg.OUTPUT, cfg.SEQLENGTH, cfg.SEQLENGTH])

confv = np.full([cfg.OUTPUT],-1,dtype="float") # to save confidences for reporting
for i in range(cfg.OUTPUT):
    rankmatrices[i][:] = rankmatrices[i].transpose()
    importance_v, confidence_v, confv[i] = sfn.get_importance_confidence(rankmatrices[i])
    if VERBOSE:
        print("label", i)
        #print(rankmatrices[i])
        print("Importance vector", importance_v)
        print("Confidence vector", confidence_v)
        print("Overall confidence", confv[i])
        print()

if VERBOSE:
    print("Confidence analysis complete.")
    print("Beginning signature generation...")

# #####################################################
# Getting signatures
# #####################################################

signatures = np.full([cfg.OUTPUT, cfg.SEQLENGTH], -1, dtype="float")

for i, list in enumerate(attentionlists):
    attentions = np.array(list)
    signatures[i][:] = np.mean(attentions, axis=0)
    if VERBOSE:
        print(signatures[i], "min: {0:.2f},".format(round(min(signatures[i]),2)), "max: {0:.2f},".format(round(max(signatures[i]),2)), "conf: {0:.3f}".format(round(np.mean(confv[i]),3)))
np.savetxt(cfg.TATTN_SIGNATURES, signatures, delimiter = ",")
if VERBOSE:
    print("Total confidence: {0:.3f}".format(round(np.mean(confv), 3)))
    print("Signature generation complete. Written to file:", os.path.basename(cfg.TATTN_SIGNATURES))
    print("Saving signature graphs to directory:", "/".join(cfg.SAVESIGSPATH.split("/")[-3:-1]) + "/")

for i in range(len(signatures)):
    y = np.array(signatures[i])
    l = len(y)
    x = np.linspace(0,l-1, num = l)

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(y[np.newaxis,:], cmap="Reds", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    ax2.set_yticks([0.0,0.5,1.0])
    ax2.set_ylim(0, 1)
    ax2.plot(x,y)

    plt.tight_layout()
    #plt.show()
    label = 0
    plt.savefig(cfg.SAVESIGSPATH + "sig_" + str(i)+".png")

if VERBOSE:
    print("Done.")
    print()
