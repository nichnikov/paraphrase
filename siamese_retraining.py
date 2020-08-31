from __future__ import absolute_import
from __future__ import print_function

import os, keras
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from keras.layers import Input, Lambda
from keras.models import load_model
import numpy as np
from keras_functions import contrastive_loss, siamese_data_prepare, accuracy
from random import shuffle

model_rout = r"./models"
data_rout = r"./data"

d2v_model = Doc2Vec.load(os.path.join(model_rout, 'bss_doc2vec_model_20200611_draft'))

paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_paraphrase.csv"))
no_paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_no_paraphrase.csv"))
df = pd.concat([paraphrase_df, no_paraphrase_df])

# df = pd.read_csv(os.path.join(data_rout, "data_for_retraining_mistakes.csv"))
print(df.shape)
n_examples = df.shape[0]
n_train = 110000

print(df.shape)
data_df = df.sample(frac=1)

print(data_df)
data_tuples = zip(list(data_df["question1"][:n_examples]), list(data_df["question2"][:n_examples]),
                  list(data_df["is_duplicate"][:n_examples]))

X, y = siamese_data_prepare(data_tuples, d2v_model, one_vector=False)

loop = 1
loops = 5
while loop <= loops:
    print(loop, "/", loops)
    X_y = list(zip(X, y))
    shuffle(X_y)
    X, y = zip(*X_y)
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
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
    x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

    x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
    x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

    keras.losses.contrastive_loss = contrastive_loss
    model = load_model(os.path.join("models", "siamese_model_d2v_nn_2020_0613.h5"))
    model.compile(loss=contrastive_loss, optimizer="Adam", metrics=[accuracy])
    print("model.compile Done")

    model.fit([x_tr1, x_tr2], tr_y,
              batch_size=1024,
              epochs=epochs,
              validation_data=([x_te1, x_te2], te_y))

    model.save(os.path.join(model_rout, 'siamese_model_d2v_nn_2020_0613.h5'))
    loop += 1
