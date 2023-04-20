#!/usr/bin/python3
"""
Simple stand alone analysis.

Authors: Alexander Katrompas
Organization: Texas State University

Usage: analysis.py filename labels
       (assuming python3 in /usr/bin/)

file input must conform to cfg.ACTUALPREDICTEDV or cfg.ACTUALPREDICTEDS
labels must be the number of categorical outputs

WARNING: script has few sanity checks. Make sure you have the right file and labels.

"""

###########################
# imports
###########################

# python libs
import os
import sys
import silence_tensorflow.auto
sys.path.insert(1, os.getcwd() + '/../src/')

# local libs
import functions as fn

argc = len(sys.argv)
labels = 0
if argc == 3 and os.path.exists(sys.argv[1]):
    try:
        labels = int(sys.argv[2])
    except:
        pass

if labels:
    fn.print_stats(sys.argv[1], labels)
