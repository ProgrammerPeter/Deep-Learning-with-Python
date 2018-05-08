# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:26:15 2018

@author: Tc
"""

import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  # All printable ASCII characters.
#token_index = dict(zip(range(1, len(characters) + 1), characters))
token_index = dict(zip(characters, range(1, len(characters) + 1)))  # We can only search values with keys.

max_length = 50
#results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.