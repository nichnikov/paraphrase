import os, pickle, json
import pandas as pd
from lingv_functions import lsi_model_create
from texts_processors import SimpleTokenizer, LsiTokenizer
from utility import Loader, models_create
from gensim.similarities import MatrixSimilarity


def model_make(**kwargs):
    if "lingv_rules_csv_path" in kwargs:
        lingv_rules_df = pd.read_csv(kwargs["lingv_rules_csv_path"])
        rules_dict = {'rules': list(lingv_rules_df["rules"]),
                      'words': list(lingv_rules_df["words"]),
                      'tags': list(lingv_rules_df["tag"]),
                      'coeff': list(lingv_rules_df["coeff"])}
    else:
        rules_dict = {'rules': [],
                      'words': [],
                      'tags': [],
                      'coeff': []}

    if "lingvo" in kwargs:
        lingvo_list = []
        for lingv_dict in kwargs["lingvo"]:
            if "stopwords_csv_path" in lingv_dict:
                stopwords = []
                for file_name in lingv_dict["stopwords_csv_path"]:
                    stopwords_df = pd.read_csv(file_name)
                    stopwords.append(list(stopwords_df['words']))
                lingvo_list.append({"stopwords": stopwords, "tokenize": True})

            if "synonyms_files_csv_path" in lingv_dict:
                synonyms = []
                for file_name in lingv_dict["synonyms_files_csv_path"]:
                    synonyms_df = pd.read_csv(file_name)
                    synonyms.append(list(zip(synonyms_df["words"], synonyms_df["initial_forms"])))
                lingvo_list.append({"synonyms": synonyms, "tokenize": True})

            if "ngrams_csv_path" in lingv_dict:
                ngrams = []
                for file_name in lingv_dict["ngrams_csv_path"]:
                    ngrams_df = pd.read_csv(file_name)
                    ngrams.append([(" ".join([w1, w2]), tk) for w1, w2, tk in
                                   zip(list(ngrams_df["w1"]), list(ngrams_df["w2"]), list(ngrams_df["bigrams"]))])
                lingvo_list.append({"ngrams": ngrams, "tokenize": False})

            if "workwords_csv_path" in lingv_dict:
                workwords = []
                for file_name in lingv_dict["workwords_csv_path"]:
                    workwords_df = pd.read_csv(file_name)
                    workwords.append(list(workwords_df['words']))
                lingvo_list.append({"workwords": workwords, "tokenize": True})

    else:
        lingvo_list = []

    if kwargs["model_type"] == 'simple_rules':
        # соберем модель для запуска токенизатора:
        model_dict_simple = models_create(tokenizer="SimpleTokenizer", model_type="simple_rules",
                                          lingv_rules=rules_dict, lingvo=lingvo_list)
        tokenizer = SimpleTokenizer(Loader(model_dict_simple))

        if "is_lingvo_lemmatize" in kwargs:
            is_lingvo_lemmatize = kwargs["is_lingvo_lemmatize"]
            if is_lingvo_lemmatize:
                # print("tokenizer.dictionaries:", "\n", tokenizer.dictionaries)
                lingvo_list = tokenizer.dictionaries
        else:
            is_lingvo_lemmatize = False

        if "is_etalons_lemmatize" in kwargs:
            is_etalons_lemmatize = kwargs["is_etalons_lemmatize"]
            if is_etalons_lemmatize:
                rules_dict["words"] = tokenizer.texts_processing(rules_dict["words"])
        else:
            is_etalons_lemmatize = False

        result_model_dict = models_create(tokenizer=kwargs["tokenizer"], model_type=kwargs["model_type"],
                                          lingv_rules=rules_dict, lingvo=lingvo_list,
                                          is_lingvo_lemmatize=is_lingvo_lemmatize,
                                          is_etalons_lemmatize=is_etalons_lemmatize)
        return result_model_dict

    if kwargs["model_type"] == 'lsi':
        # загрузка lsi модели:
        with open(kwargs["lsi_model_path"], "rb") as f:
            lsi_dict = pickle.load(f)

        # соберем модель для запуска токенизатора:
        model_dict_lsi = models_create(tokenizer="LsiTokenizer", model_type="lsi", lingv_rules=rules_dict,
                                       lingvo=lingvo_list, texts_algorithms=lsi_dict)

        tokenizer = LsiTokenizer(Loader(model_dict_lsi))
        if 'index' not in lsi_dict:
            et_vectors = tokenizer.texts_processing(rules_dict['words'])
            index = MatrixSimilarity(et_vectors, num_features=lsi_dict["num_topics"])
            lsi_dict["index"] = index

        if "is_lingvo_lemmatize" in kwargs:
            is_lingvo_lemmatize = kwargs["is_lingvo_lemmatize"]
            if is_lingvo_lemmatize:
                lingvo_list = tokenizer.dictionaries
        else:
            is_lingvo_lemmatize = False

        result_model_dict = models_create(tokenizer="LsiTokenizer", model_type=kwargs["model_type"],
                                          lingv_rules=rules_dict, lingvo=lingvo_list,
                                          texts_algorithms=lsi_dict,
                                          is_lingvo_lemmatize=is_lingvo_lemmatize,
                                          is_etalons_lemmatize=True)
        return result_model_dict


if __name__ == "__main__":
    # загрузка файлов с данными:
    tokenize_path = r'./tokenize_model'
    # models_path = r'./models'
    model_parameters = {
        "tokenizator_model": "tokenizer_parameters_simple.json"
    }

    for model_p in model_parameters:
        with open(os.path.join(tokenize_path, model_parameters[model_p]), "r") as f:
            model_param = json.load(f)
            model = model_make(**model_param)
            with open(os.path.join(tokenize_path, model_p + ".pickle"), "wb") as f:
                pickle.dump(model, f)

    print(model)
    # for cl in model["lingvo"]:
    #    print(cl)
