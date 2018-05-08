# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:28:04 2018

@author: Tc
"""

from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible token, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)