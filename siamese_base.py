from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, Input, Conv1D, Flatten, Lambda
from keras.models import Model, Sequential
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


"""
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Bidirectional(256, input_shape=input_shape)(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
"""


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(5000, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1500, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='relu')(x)
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
#  кортежей 
# if one_vector == True [(v: np.array, lb: int), ...]
# if one_vector == False [((v1: np.array, v2: np.array), lb: int), ...]

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
    model_name = 'siamese_model_d2v_lstm_2020_0613.h5'

    d2v_model = Doc2Vec.load(os.path.join(model_rout, 'bss_doc2vec_model_20200611_draft'))

    paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_paraphrase.csv"))
    no_paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_no_paraphrase.csv"))
    df = pd.concat([paraphrase_df, no_paraphrase_df])


    print(no_paraphrase_df.shape)
    print(paraphrase_df.shape)
    print(df.shape)
    # df = pd.read_csv(os.path.join(data_rout, "data_for_learning_lemm_balance.csv"))
    # df = pd.read_csv(os.path.join(data_rout, "data_for_learning_lemm.csv"))
    # print(df.shape)

    data_df = df.sample(frac=1)
    n_examples = df.shape[0]
    n_train = 55000

    print(data_df)

    data_tuples = zip(list(data_df["question1"][:n_examples]), list(data_df["question2"][:n_examples]),
                      list(data_df["is_duplicate"][:n_examples]))

    one_vector = False
    X, y = siamese_data_prepare(data_tuples, d2v_model, one_vector=one_vector)
    # print(list(data_tuples)[:10])



    # т. к. датафрейм уже перемешан, то можно получившиеся списки делить на тренировочную и тестовую выборки
    x_train = X[:n_train]
    x_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    # ===========================================================================================
    epochs = 50

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
    model.compile(loss=contrastive_loss, optimizer="Adam", metrics=[accuracy])
    print("model.compile Done")

    print("tr_pairs.shape:", tr_pairs.shape)

    if one_vector:
        x_tr1 = tr_pairs.reshape(tr_pairs.shape[0], tr_pairs.shape[1], 1)
        x_tr2 = tr_pairs.reshape(tr_pairs.shape[0], tr_pairs.shape[1], 1)

        x_te1 = te_pairs.reshape(te_pairs.shape[0], te_pairs.shape[1], 1)
        x_te2 = te_pairs.reshape(te_pairs.shape[0], te_pairs.shape[1], 1)

    else:
        x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
        x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

        x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
        x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

    print("x_tr1.shape:", x_tr1.shape)

    model.fit([x_tr1, x_tr2], tr_y,
              batch_size=512,
              epochs=epochs,
              validation_data=([x_te1, x_te2], te_y))

    model.save(os.path.join(model_rout, model_name))
