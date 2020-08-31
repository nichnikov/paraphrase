# Тестируем под задачу "быстрых ответов" первоначальные вопросы (на которых сеть обучалась) используем в качестве
# эталонов, смотрим, насколько качественно она отбирает во входящем потоке похожие на них вопросы
import os, pickle, time
from utility import Loader
from texts_processors import TokenizerApply
import pandas as pd

# загрузка файлов с данными:
tokenize_path = r'./tokenize_model'
test_path = r'./test'

with open(os.path.join(tokenize_path, "tokenizator_model.pickle"), "rb") as f:
    tokenize_model = pickle.load(f)
    tokenize_loader = Loader(tokenize_model)

tknz = TokenizerApply(tokenize_loader)

# загрузка вопросов
df_data = pd.read_csv(os.path.join(test_path, "ндс_прибыль_5000.csv"))
df_data.rename(columns={"0": "text"}, inplace=True)

# загрузка словаря, который "знает" нейронная сеть
work_dict_df = pd.read_csv(os.path.join(test_path, "dictionary_work.csv"))
work_dict_list = list(work_dict_df["token"])
print(work_dict_list)

# загрузка эталонов (первоначальных запросов, на которых обучалась нейронная сеть)
df_etalons = pd.read_csv(os.path.join(test_path, "etalons.csv"))

df_etalons = df_data
tktxs = tknz.texts_processing(df_data["text"])
tkets = tknz.texts_processing(df_etalons["text"])

txts_list = [x for x in tktxs]
tkets_list = [x for x in tkets]

# фрагменты, все слова которых "знакомы" социальным сетям
true_fragments = []
txts_tks = list(zip(list(df_data["text"]), txts_list))
etls_tks = list(zip(list(df_etalons["text"]), tkets_list))

for tx, tk in txts_tks:
    if len(set(tk) & set(work_dict_list)) == len(tk):
        true_fragments.append((tx, tk))

print("true_fragments:")
print(true_fragments[:10])
print(len(true_fragments))

print("tkets_list:")
print(etls_tks[:10])

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


models_rout = r"./models"

# load models:
d2v_model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))
print("d2v_model load Done")

keras.losses.contrastive_loss = contrastive_loss
nn_model = load_model(os.path.join(models_rout, 'siamese_model_d2v_nn_2020_0613.h5'))
print("lstm_model load Done")

# tx1 = "камеральные проверки по прибыли сроки"
#  tx1 = "в какие сроки налоговая должна отправить акт камеральной проверки после окончания"
# tx1_tk = tknz.texts_processing([tx1])[0]
# print(tx1)

results = []
k = 0
for tx1, tx1_tk in etls_tks:
    while k < 2000:
        for tx2, tx2_tk in true_fragments:
            d2v_vec1 = d2v_model.infer_vector(tx1_tk)
            d2v_vec2 = d2v_model.infer_vector(tx2_tk)
            v1 = d2v_vec1.reshape(1, 300, 1)
            v2 = d2v_vec2.reshape(1, 300, 1)
            t0 = time.time()
            score = nn_model.predict([v1, v2])
            delta_t = time.time() - t0
            if score < 0.1:
                print(k)
                results.append(("tx1:", tx1, "tx2:", tx2, score[0][0], delta_t))
                print("tx1:", tx1, "tx2:", tx2, score[0][0], delta_t)
                k += 1

results_df = pd.DataFrame(results)
print(results_df)
# results_df.to_csv(os.path.join(test_path, "results_etalons.csv"))
results_df.to_csv(os.path.join(test_path, "results_general_qsts.csv"))
