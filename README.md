# Temporal Attention Signatures for Interpretable Time-Series Prediction
# Sequence-to-Sequence Modeling with Temporal Attention

## Description
Code from the paper titled "Temporal Attention Signatures for Interpretable
Time-Series Prediction" in review with ICANN2023. Code also includes future
work in sequence-to-sequence time-series modeling for an upcoming paper on
that topic, however, the additional code does not interfere with Temporal
Attention Signature study and generation. 

## Abstract.
Deep neural networks are widely used for time-series predic-
tion due to their popularity and accuracy, but their internal workings are
often considered inscrutable. Recent progress has been made in the inter-
pretation of neural networks, particularly in models employing attention
mechanisms and feature maps for image classification. These models al-
low for the construction of feature heat maps that elucidate the salient
aspects of the data. While human observers can easily verify the impor-
tance of features in image classification, time-series data and modeling
are typically more difficult to interpret. Addressing this, we propose a
new approach that combines temporal attention, a fusion of recurrent
neural networks and attention along with attention visualization, to cre-
ate temporal attention signatures similar to image attention heat maps.
Temporal attention, in addition to achieving higher accuracy than re-
currence alone, will show that different label classes result in different
attention signatures, indicating that neural networks attend to different
portions of time-series sequences depending on what is being predicted.
We conclude by discussing practical applications of this novel approach,
including model interpretation, selection of sequence length, and model
validation. This yields more robust, higher accuracy, higher confidence,
and interpretable models.

## Setup
Requires
 - Python 3.6 or greater
 - TensorFlow 2.4 or greater
 - Compatible versions of
   - numpy
   - pandas
   - xgboost
   - sklearn
   - matplotlib

## Usage
Usage: main.py [-vg]
(assuming python3 in /usr/bin/)

v: verbose mode (optional)
g: graphing mode (optional)

Configuration settings in src/cfg.py

For temporal attention signature research and analysis
 - STEP=1, SEC2SEQ=False, CAPATTN=1,
 - SHUFFLE = True or False depending on type of data (usually True for sig generation)
 - special case, ECG data, SLIDE=True, otherwise False
 - run this file
 - run scripts/sig-process.py

For Seq2Seq research and analysis
 - STEP=0, SHUFFLE=False
 - run this file first with SEC2SEQ=False to get A/P seq2vec for analysis and comparison to seq2seq
 - then run this file with cfg.SEC2SEQ=True and CAPATTN=1 to get A/P seq2seq for analysis and comparison
 - for seq2vec to seq2seq analysis
    - run hard vote
    - run soft vote without attention
    - run soft vote with attention (if you don't need this, run with CAPATTN=0 for speed)
    - run stacking
    - compare all five
## Citation
See paper Temporal Attention Signatures for Interpretable Time-Series Prediction
