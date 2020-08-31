import os
from clickhouse_connect import questions_from_clickhouse
from lingv_functions import bigrams_dictionary_create
from models_maker import model_make
from texts_processors import SimpleTokenizer
from utility import Loader

key_words = ['%%']
questions = []
for word in key_words:
    res = questions_from_clickhouse(clickhose_host="srv02.ml.dev.msk3.sl.amedia.tech", user='nichnikov',
                                    password='CGMKRW5rNHzZAWmvyNd6C8EfR3jZDwbV', date_in='2020-01-01',
                                    date_out='2020-05-31', limit=100000, pubids_tuple=(10, 12, 16), key_word=word)

    qs, dts = zip(*res)
    questions = questions + list(qs)

data_path = r'./data'
models_path = r'./models'

model_parameters = {"model_type": "simple_rules",
                    "stopwords_csv_path": os.path.join(data_path, "04_stopwords.csv"),
                    "synonyms_files_csv_path": [os.path.join(data_path, "01_synonyms.csv"),
                                                os.path.join(data_path, "02_synonyms.csv"),
                                                os.path.join(data_path, "03_synonyms.csv")],
                    "tokenizer": "SimpleTokenizer",
                    "is_lingvo_lemmatize": True,
                    "is_etalons_lemmatize": True
                    }

model_for_tokenizer = model_make(**model_parameters)
print(model_for_tokenizer)
tokenizer = SimpleTokenizer(Loader(model_for_tokenizer))
tknz_texts = tokenizer.texts_processing(questions)

bigrams_df = bigrams_dictionary_create(tknz_texts)
print(bigrams_df)
bigrams_df.to_csv(os.path.join(data_path, "kss_ngrams_candidates.csv"))
