import os, pickle, json

# загрузка файлов с данными:
tokenize_path = r'./tokenize_model'

tokenizer_parameters_simple = {"model_type": "simple_rules",
                               "tokenizer": "SimpleTokenizer",
                               "lingvo": [{"stopwords_csv_path": [os.path.join(tokenize_path, "stopwords.csv")]},
                                          {"ngrams_csv_path": [os.path.join(tokenize_path, "bss_ngrams.csv")]},
                                          {"synonyms_files_csv_path": [os.path.join(tokenize_path, "synonyms.csv")]}],
                               "is_lingvo_lemmatize": True,
                               "is_etalons_lemmatize": True
                               }

with open(os.path.join(tokenize_path, "tokenizer_parameters_simple.json"), "w") as f:
    json.dump(tokenizer_parameters_simple, f)
