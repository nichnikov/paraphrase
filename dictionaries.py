# создание словаря по коллекции текстов с использованием токенизатора модели
import os, pickle
import pandas as pd
from texts_processors import TokenizerApply
from utility import Loader
from lingv_functions import frequency_dictionary_create
from gensim.models.doc2vec import Doc2Vec
from utility import splitArray
from collections import deque


# функция, лемматизирующая большие коллекции текстов
# записывает их в указанный файл  csv:
def big_collections_lingv_handle(text_list, tk_appl, butch_size=10000,
                                 csv_file_rout=os.path.join('./data', "камеральная_проверка_вопросы_лемм.csv")):
    spl_qsts = splitArray(text_list, butch_size)
    q = deque(spl_qsts)
    k=1
    n=len(spl_qsts)
    while q:
        print(k, '/', n)
        k += 1
        lemm_quests = tk_appl.texts_processing(q.pop())
        lemm_quests_df = pd.DataFrame([" ".join(x) for x in lemm_quests])
        lemm_quests_df.to_csv(csv_file_rout, mode='a', header=False)
    return 0


data_rout = r'./data'
models_rout = r'./models'

# частотный словарь слов обрабатываемой коллекции текстов:
"""
data_df = pd.read_csv(os.path.join(data_rout, "data_group_01.csv"))
with open(os.path.join(models_rout, "tokenizator_model.pickle"), "br") as f:
    model = pickle.load(f)

tk_appl = TokenizerApply(Loader(model))
lemm_txts = tk_appl.texts_processing(data_df["text"])
frq_dct = frequency_dictionary_create(lemm_txts)
frq_dct.to_csv(os.path.join(data_rout, "dictionary_work.csv"))
"""

# словарь doc2vec модели:
d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))
d2v_dict = [tk for tk in d2v_model.wv.vocab]
d2v_dict_df = pd.DataFrame(d2v_dict)
d2v_dict_df.to_csv(os.path.join(data_rout, "dictionary_d2v_20200611_draft.csv"))


# словарь коллекции для обучения doc2vec модели:
# (1) лемматизация текстов:
# questions_df = pd.read_csv(os.path.join(data_rout, "камеральная_проверка_вопросы.csv"))
# quests_list = list(questions_df["text"])
# big_collections_lingv_handle(quests_list)

# (2) построение частотного словаря:
"""
lemm_questions_df = pd.read_csv(os.path.join(data_rout, "камеральная_проверка_вопросы_лемм.csv"))
print(lemm_questions_df)
lm_q = [x.split() for x in list(lemm_questions_df["text"])]
print(lm_q[:10])
ques_frq_dct = frequency_dictionary_create(lm_q)
print(ques_frq_dct)
ques_frq_dct.to_csv(os.path.join(data_rout, "dictionary_quests_big.csv"))"""