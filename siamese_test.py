from __future__ import absolute_import
from __future__ import print_function

# нужно добавить проверку на наличие слов входящих запросов в словаре (например, в словах из правил)
import pickle, os
import pandas as pd
from keras.models import load_model
from keras import backend as K
import keras.losses
from gensim.models.doc2vec import Doc2Vec


def contrastive_loss(y_true, y_pred):
    # Contrastive loss from Hadsell-et-al.'06
    # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# для офисного компьютера:
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
'''


def binarizator(x, coeff):
    if x > coeff:
        return 1
    else:
        return 0


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def nn_testing(df, nn_model, d2v_model):
    test_results = []
    for q1, q2, lb in zip(list(df["question1"]), list(df["question2"]), list(df["is_duplicate"])):
        d2v_vec1 = d2v_model.infer_vector(q1.split())
        d2v_vec2 = d2v_model.infer_vector(q2.split())
        v1 = d2v_vec1.reshape(1, 300, 1)
        v2 = d2v_vec2.reshape(1, 300, 1)
        score = nn_model.predict([v1, v2])
        test_results.append((q1, q2, score[0][0], lb))
    return test_results


data_rout = r"./data"
models_rout = r"./models"

# load models:
d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))
print("d2v_model load Done")

keras.losses.contrastive_loss = contrastive_loss
nn_model = load_model(os.path.join(models_rout, 'siamese_model_d2v_nn_2020_0613.h5'))
print("lstm_model load Done")

# load data:
# quests_df = pd.read_csv(os.path.join(data_rout, 'data_for_learning_lemm.csv'))
paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_paraphrase.csv"))
no_paraphrase_df = pd.read_csv(os.path.join(data_rout, "b_no_paraphrase.csv"))
quests_df = pd.concat([paraphrase_df, no_paraphrase_df])

test_results = nn_testing(quests_df, nn_model, d2v_model)
print(test_results[:10])
test_results_df = pd.DataFrame(test_results, columns=["qt1", "qt2", "score", "true_lb"])
print(test_results_df)

test_results_df.to_csv(os.path.join(data_rout, "test_systems_predict.csv"), decimal=",")
