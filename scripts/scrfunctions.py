"""
Supporting stand-alone functions for scripts
This file does not run on it's own, it's supporting functions only.

Authors: Alexander Katrompas
Organization: Texas State University

"""
import os
import sys
import numpy as np
from os import getcwd

sys.path.insert(1, getcwd() + '/../src/')
import cfg

def get_importance_confidence(rankmatrix):
    """
    turn a rank matrix for a label into an importance vector

    @param (numpy array) rankmatrix : cfg.SEQLENGTH x cfg.SEQLENGTH of ranks

    Return: importance vector, confidence vector, average confidence
    """

    importancev = np.zeros([cfg.SEQLENGTH], dtype=int)
    confindencev = np.zeros([cfg.SEQLENGTH], dtype=float)
    sums = np.zeros([cfg.SEQLENGTH], dtype=int)

    for i in range(cfg.SEQLENGTH):
        importancev[i] = np.argmax(rankmatrix[i])
        sums[i] = sum(rankmatrix[i])

    for i in range(cfg.SEQLENGTH):
        # confidencev is the percent the position occurs at the highest position only
        # i.e. it's how much the highest position is the highest position
        confindencev[i] = round(rankmatrix[i][importancev[i]] / sums[i] , 3)

    # sanity check
    if np.min(sums) != np.max(sums):
        raise("error in getSignature")

    return importancev, confindencev, round(sum(confindencev)/cfg.SEQLENGTH,3)


def get_ranks(attentionv):
    """
    takes a sequence of normalized attention and calculates the importance ranking

    @param (numpy array) attentionv : cfg.SEQLENGTH of attention scores

    Return: rank vector of attention scores
    """

    temp = attentionv.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(cfg.SEQLENGTH)

    # fix in case multiple 0s or 1s in normalized array
    # very, very rare case and doesn't statistically affect analysis
    for i in range(cfg.SEQLENGTH):
        if attentionv[i] < .01:
            ranks[i] = 0
        elif attentionv[i] > .995:
            ranks[i] = cfg.SEQLENGTH - 1
    return ranks

def get_actual_vec(aseqs):
    """
    converts seq2seq actual values back into the vector from which they came

    @param (numpy 2D) aseqs : seq2seq actual values

    Return: seq2vec actual values (matches what's in the original data file
    """

    length = aseqs.shape[0]-cfg.SEQLENGTH+1
    avec = np.full((length), -1)
    for i in range(length):
        avec[i] = aseqs[i][cfg.SEQLENGTH-1]
    return avec

def hardvote(vec):
    """
    takes a hard vote from each time-step's prediction for a particular time-step

    @param (numpy 1D) vec : pre-processed vector of the column of predictions

    Return: the most common value which is the "vote" of all the time-steps
    """

    # TODO what if they are equal?
    # ignore it since it's a rare case and soft voting solves it?

    # get values and corresponding counts
    values, counts = np.unique(vec, return_counts=True)
    most_common_value = values[np.argmax(counts)]
    #print(i+1, vec)
    #print(counts, values)
    #print(most_common_value)
    return most_common_value

