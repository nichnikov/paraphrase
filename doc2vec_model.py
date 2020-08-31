import os, pickle
from collections import deque
from texts_processors import TokenizerApply
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from utility import splitArray, Loader
from clickhouse_connect import  questions_from_clickhouse


def create_doc2vec_model(**kwargs):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # внимание! очень важно, как разбиты токены в split_txt
    # вид должен быть следующий: [['word1', 'word2', ...], ['word1', 'word5', ...], ... ]
    tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(kwargs["split_txt"])]
    print("tagged_data made, example:", tagged_data[0])
    model = Doc2Vec(tagged_data, vector_size=300, window=3, min_count=1, workers=8)
    model.build_vocab(documents=tagged_data, update=True)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=25)
    model.save(kwargs["model_rout"])
    print("model made and saved")
    return 0


def update_doc2vec_model(**kwargs):
    # внимание! очень важно, как разбиты токены в tokens_texts
    # вид должен быть следующий: [['word1', 'word2', ...], ['word1', 'word5', ...], ... ]
    # split_txts дополнительно разбивается на указанное количество кусков, из которых образуется очередь (если текстов дял обучения модели слишком много)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Doc2Vec.load(kwargs["initial_model_rout"])
    model.save(kwargs["updated_model_rout"])
    print("model loaded and saved in new path")

    # разбиваем входящие тексты из токенов на куски для формирования очереди (указываем размер)
    split_txts = splitArray(kwargs["tokens_texts"], kwargs["chunk_size"])

    q = deque(split_txts)
    # счетчик
    k = 1
    n = len(split_txts)
    print("deque length is:", n)
    while q:
        print(k, '/', n)

        # дообучение модели Doc2Vec по оставшимся текстам:
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(q.pop())]
        model = Doc2Vec.load(kwargs["updated_model_rout"])
        print("loop:", k, "model loaded")
        model.build_vocab(documents=documents, update=True)
        print("loop:", k, "build_vocab done")
        try:
            model.train(documents, total_examples=model.corpus_count, epochs=25)
            print(documents[:10])
        except Exception:
            print("ModeLearningErr")
        model.save(kwargs["updated_model_rout"])
        k += 1
    return 0


if __name__ == "__main__":
    data_rout = r'./data'
    models_rout = r'./models'

    # подготовка данных на основании синонимов и биграмм
    """
    res = questions_from_clickhouse(clickhose_host="srv02.ml.dev.msk3.sl.amedia.tech", user='nichnikov',
                                    password='CGMKRW5rNHzZAWmvyNd6C8EfR3jZDwbV', date_in='2017-01-01',
                                    date_out='2020-05-31', limit=1000000, pubids_tuple=(6, 8, 9), key_word="%2019год %")
    qs, dts = zip(*res)
    print(qs)
    print(len(qs))

    # загрузим токенизатор
    with open(os.path.join(models_rout, "tokenizator_model.pickle"), "br") as f:
        model = pickle.load(f)

    tk_appl = TokenizerApply(Loader(model))
    lemm_txts = tk_appl.texts_processing(qs[:1000])
    print(lemm_txts)

    data_df = pd.read_csv(os.path.join(data_rout, "камеральная_проверка_вопросы_лемм.csv"))
    texts_list_split = [x.split() for x in list(data_df["text"])]

    create_doc2vec_model(split_txt=texts_list_split[:1000], model_rout=os.path.join(models_rout, "bss_doc2vec_model_20200611_draft"))

    update_doc2vec_model(tokens_texts=texts_list_split[1000:], initial_model_rout=os.path.join(models_rout, "bss_doc2vec_model_20200611_draft"),
                         updated_model_rout=os.path.join(models_rout,"bss_doc2vec_model_20200611_draft"), chunk_size=1000)
    """
    model = Doc2Vec.load(os.path.join(models_rout, 'bss_doc2vec_model_20200611_draft'))

    # test ['срок', 'камеральныйпроверка', 'декларация', 'налогприбыль']] [['срок', 'камеральныйпроверка', '3ндфл']]
    test_data = ['налогприбыль']
    v = model.infer_vector(test_data)
    # print(v)

    # print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model.most_similar(positive=test_data, topn=50))
