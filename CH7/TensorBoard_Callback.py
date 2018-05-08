# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:17:38 2018

@author: Tc
"""

import keras 
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 2000  # number of words to consider as features
max_len = 500  # cut texts after this number of words (among top max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

callbacks = [
        keras.callbacks.TensorBoard(
                # Log files will be written at this location
                log_dir='my_log_dir',
                # We will record activation histograms every 1 epoch
                histogram_freq=1,
                # We will record embedding data every 1 epoch
                embeddings_freq=1,
        )
]
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)


