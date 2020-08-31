from __future__ import absolute_import
from __future__ import print_function

import os # , io, pickle, re, glob, argparse, random
# from collections import deque
import pandas as pd
# import numpy as np
# from random import shuffle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from pymystem3 import Mystem
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, Input, Conv1D, Flatten, Lambda
from keras.models import Model, Sequential
# import tensorflow as tf
# from keras.preprocessing import sequence
# from keras.datasets import imdb
# from keras.optimizers import SGD, Adam, RMSprop, Adamax
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.layers import Layer
from keras.models import load_model

import numpy as np

from keras.optimizers import RMSprop
from keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    # x = Flatten()(input)
    x = LSTM(250, input_shape=input_shape)(input)
    x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# функция векторизации, возвращает список:
def siamese_data_prepare(data_tuples, d2v_model, one_vector=True):
    X, y = [], []
    if one_vector:
        for q1, q2, lb in data_tuples:
            tx_list = q1.split() + q2.split()
            v = d2v_model.infer_vector(tx_list)
            X.append(v)
            y.append(lb)
    else:
        for q1, q2, lb in data_tuples:
            v1 = d2v_model.infer_vector(q1.split())
            v2 = d2v_model.infer_vector(q2.split())
            X.append((v1, v2))
            y.append(lb)
    return X, y


if __name__ == "__main__":
    model_rout = r"./models"
    data_rout = r"./data"

    d2v_model = Doc2Vec.load(os.path.join(model_rout, 'bss_doc2vec_model'))
    df = pd.read_csv(os.path.join(data_rout, "data_for_learning_lemm.tsv"), sep='\t')

    print(df.shape)
    data_df = df.sample(frac=1)

    n_examples = 5763
    n_train = 5000

    data_tuples = zip(list(data_df["question1"][:n_examples]), list(data_df["question2"][:n_examples]),
                      list(data_df["is_duplicate"][:n_examples]))

    X, y = siamese_data_prepare(data_tuples, d2v_model, one_vector=False)

    # т. к. датафрейм уже перемешан, то можно получившиеся списки делить на тренировочную и тестовую выборки
    x_train = X[:n_train]
    x_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    # ===========================================================================================

    # num_classes = 10
    epochs = 5

    tr_pairs = np.array(x_train)
    tr_y = np.array(y_train)

    te_pairs = np.array(x_test)
    te_y = np.array(y_test)

    input_shape = (300, 1)
    print(input_shape)

    # network definition
    base_network = create_base_network(input_shape)
    print("base_network Done")

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    print("processed_a, processed_b Done")

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    print("distance Done")

    model = Model([input_a, input_b], distance)
    print("model Done")

    # train
    # rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer="Adam", metrics=[accuracy])
    print("model.compile Done")

    # model = load_model(os.path.join("models", "siamese_model_d2v_lstm.h5"))

    x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
    x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

    x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
    x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

    model.fit([x_tr1, x_tr2], tr_y,
              batch_size=1024,
              epochs=epochs,
              validation_data=([x_te1, x_te2], te_y))

    model.save(os.path.join(model_rout, 'test_siamese_model_d2v_lstm.h5'))
