import os, pickle
import pandas as pd
from lingv_functions import lsi_model_create
from texts_processors import SimpleTokenizer
from utility import Loader
from clickhouse_connect import questions_from_clickhouse
from models_maker import model_make
from random import shuffle


def lsi_model_maker(**kwargs):
    key_words = ['%%']
    questions = []
    for word in key_words:
        res = questions_from_clickhouse(clickhose_host="srv02.ml.dev.msk3.sl.amedia.tech", user='nichnikov',
                                        password='CGMKRW5rNHzZAWmvyNd6C8EfR3jZDwbV', date_in='2020-01-01',
                                        date_out='2020-05-31', limit=100000, pubids_tuple=kwargs["pubids_tuple"],
                                        key_word=word)

        qs, dts = zip(*res)
        questions = questions + list(qs)

    shuffle(questions)
    etalons_df = pd.read_csv(kwargs["lingv_rules_csv_path"])
    data_for_models = list(etalons_df["words"]) + questions[:100000]
    print(data_for_models[:10])
    print(len(data_for_models))

    # модель для токенизатора:
    model_parameters = {"model_type": "simple_rules",
                        "stopwords_csv_path": os.path.join(data_path, "04_stopwords.csv"),
                        "ngrams_csv_path": os.path.join(data_path, "kss_ngrams.csv"),
                        "synonyms_files_csv_path": [os.path.join(data_path, "01_synonyms.csv"),
                                                    os.path.join(data_path, "02_synonyms.csv"),
                                                    os.path.join(data_path, "03_synonyms.csv")],
                        "tokenizer": "SimpleTokenizer",
                        "is_lingvo_lemmatize": True,
                        "is_etalons_lemmatize": True
                        }

    model_for_tokenizer = model_make(**model_parameters)
    tokenizer = SimpleTokenizer(Loader(model_for_tokenizer))

    tz_txs = tokenizer.texts_processing(data_for_models)

    # соберем LSI модель на основании коллекции из 100 тысяч вопросов:
    lsi_model_dict = lsi_model_create(tz_txs, topics=1500)

    with open(kwargs["lsi_model_path"], "bw") as f:
        pickle.dump(lsi_model_dict, f)

    return 0


# загрузка файлов с данными:
# data_rout = r'./data'
# models_rout = r'./models'
data_path = r'./data'
models_path = r'./models'


kss_pubids_tuple = (10, 12, 16)
bss_pubids_tuple = (6, 8, 9)

kss_lsi_parameters = {
    "lingv_rules_csv_path": os.path.join(data_path, "kss_lingv_rules_lsi.csv"),
    "pubids_tuple": (10, 12, 16),
    "lsi_model_path": os.path.join(data_path, "kss_lsi_model_parameters.pickle")}

bss_lsi_parameters = {
    "lingv_rules_csv_path": os.path.join(data_path, "bss_lingv_rules_lsi.csv"),
    "pubids_tuple": (6, 8, 9),
    "lsi_model_path": os.path.join(data_path, "bss_lsi_model_parameters.pickle")}

lsi_model_maker(**bss_lsi_parameters)