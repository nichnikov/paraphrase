# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:51:41 2018

@author: alexey
"""

'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).

https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.p
'''
# %%
# from __future__ import print_function
import keras
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
import os

'''
def cnn_constructor(x_train, num_classes):
    model = Sequential()
    model.add(Conv1D(10, kernel_size = 3, strides=1, input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model 
'''


def cnn_constructor(x_train, num_classes):
    model = Sequential()
    model.add(Conv1D(12, kernel_size=6, strides=1, input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


data_rout = r'./data'
save_dir = r'./models'

tensors = np.load(os.path.join(data_rout, 'tensors_for_nn.npy'))
lables = np.load(os.path.join(data_rout, 'lables_for_nn.npy'))

uniqueValues, indicesList = np.unique(lables, return_index=True)
print(uniqueValues, indicesList)

print(list(lables).count(1), list(lables).count(2), list(lables).count(3))

print(tensors.shape)
print(lables.shape)

tnsrs_lbls = list(zip(list(tensors), list(lables.reshape(lables.shape[0], 1))))
shuffle(tnsrs_lbls)

# %%
tensors, lables = zip(*tnsrs_lbls)
x_train = np.array(tensors)
y_train = np.array(lables)

print(x_train.shape)
print(y_train.shape)

# %%
# initiate RMSprop optimizer
batch_size = 1024
num_classes = 2
epochs = 100
data_augmentation = True
num_predictions = 20
model_name = 'keras_cnn_test_model_balanced3.h5'

# пронумеруем классы по феншую (нумерация классов начинается с нуля)
np.place(y_train, y_train == 1, [0])
np.place(y_train, y_train == 2, [0])
np.place(y_train, y_train == 3, [1])

# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print('train samples:', x_train.shape[0])

# Convert class vectors to binary class matrices.
x_train = x_train.astype('float32')  # x_train.reshape(4000,200, 60).astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
print(x_train.shape)
print(y_train.shape)

# %%
model = cnn_constructor(x_train, 2)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3,
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# %%
