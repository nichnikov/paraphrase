'''
нужно взять приличную выборку размеченных текстов (от 1000 до 100 000), прогнать их через обученную нейронную сеть, 
сравнить проставленный нейронной сетью класс с разметкой, выбрать расхождения и их проанализировать
отдельно отобрать акты, в которых не удалось выделить резолютивную часть (после слова "решил:")

1) нужно сохранить номера актов
2) создать механизм векторизации
'''

import os, io, pickle, re, glob, argparse
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pymystem3 import Mystem
from keras import models
from texts_processors import TokenizerApply
from utility import Loader


def list_split(txt_l, length, stride):
    return [txt_l[i:i + length] for i in range(0, len(txt_l), stride) if len(txt_l[i:i + length])]


# итератор, преобразующий отрывок в набор векторов (слово => вектор)
def iter_w2v(doc, d2v):
    for w in doc:
        try:
            vec = np.array(d2v.infer_vector([w]))
        except:
            vec = np.zeros(300)
        yield vec


# итератор, преобразующий массив документов (список, элементы которого - тексты) в массив тензоров
# возвращает тензор, соответствующий документу
# lem_doc - лемматизированный документ
def fragment_tensor_prepare(lem_doc, d2v, d2v_vector_size=300, fragment_length=15):
    doc_tensor = np.zeros([1, d2v_vector_size])
    it = iter_w2v(lem_doc[:fragment_length], d2v)
    for v in it:
        doc_tensor = np.concatenate((doc_tensor, np.array([v])), axis=0)
    doc_tensor = np.delete(doc_tensor, (0), axis=0)
    if doc_tensor.shape[0] < fragment_length:
        add_v = np.array((fragment_length - doc_tensor.shape[0]) * [[0] * d2v_vector_size])
        doc_tensor = np.concatenate((doc_tensor, add_v), axis=0)
    return np.array(doc_tensor)


data_rout = r'./data'
models_rout = r'./models'
d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))
cnn_model = models.load_model(os.path.join(models_rout, 'keras_cnn_test_model_balanced3.h5'))

# займемся вектаризацией:
# отберем фрагмент:
with open(os.path.join(models_rout, "tokenizator_model.pickle"), "br") as f:
    lingv_model = pickle.load(f)

tk_appl = TokenizerApply(Loader(lingv_model))

# tx = 'срок камеральной проверки по 2 ндфл срок камеральной проверки декларации по прибыли с убытком'
tx = ''
frg_lemm = tk_appl.texts_processing([tx])
print(frg_lemm)
fr_tensor = fragment_tensor_prepare(frg_lemm, d2v_model, d2v_vector_size=300, fragment_length=30)

result = cnn_model.predict(fr_tensor.reshape(1, 30, 300))
print(round(result[0][0], 3))

