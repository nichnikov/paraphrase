'''
берем тексты, с замененными выражениям (после ner обработки)
лемматизируем их и создаем doc2vec модель, дообучая ее на остальных текстах
'''
# %%
import os, io, pickle, re, glob, argparse
import pandas as pd
import numpy as np
from random import shuffle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pymystem3 import Mystem
from multiprocessing import Pool
import logging


def list_split(txt_l, length, stride):
    return [txt_l[i:i + length] for i in range(0, len(txt_l), stride) if len(txt_l[i:i + length]) == length]


data_rout = r'./data'
models_rout = r'./models'
texts_rout = r'/home/an/Dropbox/data/acts'
texts_for_handling_rout = r'./data/short_texts'
ner_txts_rout = r'./data/txts_ner'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
def fragment_tensor_prepare(lem_doc, d2v, d2v_vector_size=250, fragment_length=15):
    doc_tensor = np.zeros([1, d2v_vector_size])
    it = iter_w2v(lem_doc[:fragment_length], d2v)
    for v in it:
        doc_tensor = np.concatenate((doc_tensor, np.array([v])), axis=0)
    doc_tensor = np.delete(doc_tensor, (0), axis=0)
    if doc_tensor.shape[0] < fragment_length:
        add_v = np.array((fragment_length - doc_tensor.shape[0]) * [[0] * d2v_vector_size])
        doc_tensor = np.concatenate((doc_tensor, add_v), axis=0)
    return np.array(doc_tensor)


# функция, возвращащая массив тензоров (numpy array) годный для дальнейшего использования
# при обучении модели
def tensor_prepare(lem_fragments, d2v, d2v_vector_size=300, fragment_length=50):
    tensors_arr = np.zeros([1, fragment_length, d2v_vector_size])
    k = 0
    for fragment in lem_fragments:
        print(k)
        fragm_tensor = fragment_tensor_prepare(fragment, d2v, d2v_vector_size, fragment_length)
        tensors_arr = np.concatenate((tensors_arr, fragm_tensor.reshape(1, fragment_length, d2v_vector_size)), axis=0)
        k += 1
    tensors_arr = np.delete(tensors_arr, (0), axis=0)
    return tensors_arr


d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))

df = pd.read_csv(os.path.join(data_rout, "data_for_learning_lemm.csv"))
print(df.shape)

data_df = df.sample(frac=1)
texts = [x.split() + y.split() for x, y in zip(list(data_df["question1"]), list(data_df["question2"]))]

fragments_np_arr = tensor_prepare(texts, d2v_model, fragment_length=30)
print(fragments_np_arr.shape)
lables_np_arr = np.array(data_df['is_duplicate'])

print(fragments_np_arr.shape)
print(lables_np_arr.shape)

np.save(r'./data/tensors_for_nn', fragments_np_arr)
np.save(r'./data/lables_for_nn', lables_np_arr)
