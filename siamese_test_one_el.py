from __future__ import absolute_import
from __future__ import print_function

# нужно добавить проверку на наличие слов входящих запросов в словаре (например, в словах из правил)
import os, pickle
from keras.models import load_model
from keras import backend as K
import keras.losses
from keras_functions import contrastive_loss, siamese_data_prepare, accuracy
from gensim.models.doc2vec import Doc2Vec
from texts_processors import TokenizerApply
from utility import Loader
import pandas as pd

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


data_rout = r"./data/lingvo_test"
models_rout = r"./models"

# load models:
d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))
print("d2v_model load Done")

keras.losses.contrastive_loss = contrastive_loss
lstm_model = load_model(os.path.join(models_rout, 'siamese_model_d2v_nn_2020_0612.h5'))
print("lstm_model load Done")


with open(os.path.join(models_rout, "tokenizator_model.pickle"), "br") as f:
    lingv_model = pickle.load(f)

tk_appl = TokenizerApply(Loader(lingv_model))

tx1 = "сдавать ндс"
tx2 = "сдавать ндфл"
# tx1 = 'срок камеральной проверки по ндс заявленной к вычету'
# tx2 = 'срок камеральной проверке по ндс'

ts1 = tk_appl.texts_processing([tx1])
ts2 = tk_appl.texts_processing([tx2])

print(ts1, ts2)
for t1 in ts1:
    for t2 in ts2:
        d2v_vec1 = d2v_model.infer_vector(ts1[0])
        d2v_vec2 = d2v_model.infer_vector(ts2[0])
        v1 = d2v_vec1.reshape(1, 300, 1)
        v2 = d2v_vec2.reshape(1, 300, 1)
        score1 = lstm_model.predict([v1, v2])
        score2 = lstm_model.predict([v2, v1])
        print(score1, score2)
