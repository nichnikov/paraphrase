# задача теста проверить, можно ли дообучать сеть без потери качества
# сначала обучим все примеры класса 1 (4000) + 4000 первых примеров класса 0
# оценим качество на выборке из 8000 примеров (4000 класса 1 и 4000 класса 0)
# затем возьмем опять 4000 класса 1 и 4000 других класса 0 и дообучим сеть на них
# а дальше проверим качество на первой выборке, если оно не ухудшится, значит можно дообучать
# если ухудшится, значит с дообучением проблемы
from __future__ import absolute_import
from __future__ import print_function

import os, keras
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from keras.models import load_model, Model
from keras.layers import Input, Lambda
import numpy as np
from keras_functions import contrastive_loss, siamese_data_prepare, accuracy
from random import shuffle
from siamese_base import create_base_network, euclidean_distance, eucl_dist_output_shape


def data_for_nn(data_df, train_share, d2v_model):
    print(data_df)
    n_examples = data_df.shape[0]
    n_train = (train_share * 100) * n_examples // 100
    print("n_train:", n_train)
    print("n_test:", n_examples - n_train)

    data_df_shuffle = data_df.sample(frac=1)
    data_tuples = zip(list(data_df_shuffle["question1"][:n_examples]), list(data_df_shuffle["question2"][:n_examples]),
                      list(data_df_shuffle["is_duplicate"][:n_examples]))
    X, y = siamese_data_prepare(data_tuples, d2v_model, one_vector=False)
    return X, y


def base_train(**kwargs):
    # network definition
    base_network = create_base_network(kwargs['input_shape'])
    print("base_network Done")

    input_a = Input(shape=kwargs['input_shape'])
    input_b = Input(shape=kwargs['input_shape'])

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

    d2v_model = Doc2Vec.load(kwargs['d2v_model_path'])
    print("d2v_model loaded")

    X, y = data_for_nn(kwargs['df'], kwargs['train_share'], d2v_model)
    print("X, y prepared")

    n_train = int((kwargs['train_share'] * 100) * kwargs['df'].shape[0] // 100)

    X_y = list(zip(X, y))
    shuffle(X_y)
    X, y = zip(*X_y)
    # т. к. датафрейм уже перемешан, то можно получившиеся списки делить на тренировочную и тестовую выборки
    x_train = X[:n_train]
    x_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    epochs = kwargs['epochs']
    tr_pairs = np.array(x_train)
    tr_y = np.array(y_train)
    te_pairs = np.array(x_test)
    te_y = np.array(y_test)
    print("tr_pairs.shape:", tr_pairs.shape)

    x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
    x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

    x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
    x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

    print("x_tr1.shape:", x_tr1.shape)

    model.fit([x_tr1, x_tr2], tr_y,
              batch_size=512,
              epochs=epochs,
              validation_data=([x_te1, x_te2], te_y))

    return model


def retraining(**kwargs):
    d2v_model = Doc2Vec.load(kwargs['d2v_model_path'])
    print("d2v_model loaded")
    X, y = data_for_nn(kwargs['df'], kwargs['train_share'], d2v_model)
    loop = 1
    loops = kwargs['loops']
    n_train = int((kwargs['train_share'] * 100) * kwargs['df'].shape[0] // 100)
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

        epochs = kwargs['epochs']
        tr_pairs = np.array(x_train)
        tr_y = np.array(y_train)
        te_pairs = np.array(x_test)
        te_y = np.array(y_test)

        # network definition
        x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
        x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

        x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
        x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

        keras.losses.contrastive_loss = contrastive_loss
        model = load_model(kwargs["model_path"])
        model.compile(loss=contrastive_loss, optimizer="Adam", metrics=[accuracy])
        print("model.compile Done")

        model.fit([x_tr1, x_tr2], tr_y,
                  batch_size=1024,
                  epochs=epochs,
                  validation_data=([x_te1, x_te2], te_y))
        loop += 1
    return model


def nn_testing(df, nn_model_path, d2v_model_path):
    test_results = []
    d2v_model = Doc2Vec.load(d2v_model_path)
    keras.losses.contrastive_loss = contrastive_loss
    nn_model = load_model(nn_model_path)
    for q1, q2, lb in zip(list(df["question1"]), list(df["question2"]), list(df["is_duplicate"])):
        d2v_vec1 = d2v_model.infer_vector(q1.split())
        d2v_vec2 = d2v_model.infer_vector(q2.split())
        v1 = d2v_vec1.reshape(1, 300, 1)
        v2 = d2v_vec2.reshape(1, 300, 1)
        score = nn_model.predict([v1, v2])
        test_results.append((q1, q2, score[0][0], lb))
    return test_results


def binarizator(x, coeff=0.5):
    if x < coeff:
        return 1
    else:
        return 0


if __name__ == "__main__":
    model_rout = r"./models"
    data_rout = r"./data"

    paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_paraphrase.csv"))
    no_paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_no_paraphrase.csv"))
    paraphrase_df_shape = paraphrase_df.sample(frac=1)
    no_paraphrase_df_shape = no_paraphrase_df.sample(frac=1)
    df_1 = pd.concat([paraphrase_df[:3000], no_paraphrase_df[:100000]])
    df_2 = pd.concat([paraphrase_df[3000:], no_paraphrase_df[100000:]])
    print("df_1.shape:", df_1.shape)
    print("df_2.shape:", df_2.shape)

    d2v_model_path = os.path.join("models", "bss_doc2vec_model_20200611_draft")
    nn_model_path = os.path.join(model_rout, 'siamese_model_d2v_nn_2020_0614_test.h5')

    base_model = base_train(input_shape=(300, 1), df=df_1, train_share=0.8, d2v_model_path=d2v_model_path, epochs=50)
    base_model.save(nn_model_path)

    retrain_model = retraining(df=df_1, epochs=50, train_share=0.8, d2v_model_path=d2v_model_path,
                               loops=3, model_path=nn_model_path)
    retrain_model.save(nn_model_path)
    
    # осталось провести тестирование и сравнить качество на df_1 после первого и после второго обучения
    test_results = nn_testing(df_1, nn_model_path=nn_model_path, d2v_model_path=d2v_model_path)
    test_results_df = pd.DataFrame(test_results, columns=["qt1", "qt2", "score", "true_lb"])
    test_results_df["predict"] = test_results_df["score"].apply(binarizator)
    test_results_df["mistakes"] = test_results_df["predict"] - test_results_df["true_lb"]
    print(test_results_df[test_results_df["mistakes"] != 1].shape[0])
    print("accuracy:", test_results_df[test_results_df["mistakes"] != 1].shape[0] / test_results_df.shape[0])

    test_results = nn_testing(df_2, nn_model_path=nn_model_path, d2v_model_path=d2v_model_path)
    test_results_df = pd.DataFrame(test_results, columns=["qt1", "qt2", "score", "true_lb"])
    test_results_df["predict"] = test_results_df["score"].apply(binarizator)
    test_results_df["mistakes"] = test_results_df["predict"] - test_results_df["true_lb"]
    print(test_results_df[test_results_df["mistakes"] != 1].shape[0])
    print("accuracy:", test_results_df[test_results_df["mistakes"] != 1].shape[0] / test_results_df.shape[0])
