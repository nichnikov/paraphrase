import os, json
from clickhouse_driver import Client
from datetime import datetime
import pandas as pd


# превращает текстовый формат даты-времени в таймштамп для таблиц clickhouse
def date_str2timestamp(date_string="23-03-2020 22:50:47"):
    element = datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    return datetime.timestamp(element)


# функция, получающая запросы из кликхауса
# извлечение вопросов из базы по ключевым словам
# возвращает тексты запросов и даты:
def questions_from_clickhouse(**kwargs):
    client = Client(host=kwargs["clickhose_host"], user=kwargs["user"], password=kwargs["password"], )
    qsts_date = client.execute("SELECT Text, Date FROM WTree2.SearchRequest WHERE PubId IN %(pubids)s  AND Text LIKE "
                               "(%(key_word)s) AND   Date BETWEEN toDate(%(datein)s) AND toDate(%(dateout)s) "
                               "LIMIT %(limit)s", {'datein': kwargs["date_in"], 'dateout': kwargs["date_out"],
                                                   'limit': kwargs["limit"], 'pubids': kwargs["pubids_tuple"],
                                                   'key_word': kwargs["key_word"]})
    return qsts_date


if __name__ == "__main__":
    test_path = "./test"
    # key_words = ['%камеральн%', '%срок%', '%проведен%']
    # key_words = ['%%']
    key_words = ['%ндс%', '%приб%']
    questions = []
    for word in key_words:
        res = questions_from_clickhouse(clickhose_host="srv02.ml.dev.msk3.sl.amedia.tech", user='nichnikov',
                                        password='CGMKRW5rNHzZAWmvyNd6C8EfR3jZDwbV', date_in='2017-01-01',
                                        date_out='2020-05-31', limit=200000, pubids_tuple=(6, 8, 9), key_word=word)
        qs, dts = zip(*res)
        questions = questions + list(qs)

    print(len(questions))
    print(questions[:10])

    """
    # отберем те вопросы, в которых все 3 слова:
    import re
    all_keyes_list = []
    for q in questions:
        if re.findall('камеральн', q):
            if re.findall('срок', q):
                if re.findall('проведен', q):
                    all_keyes_list.append(q)
    print(len(all_keyes_list))
    questions_df = pd.DataFrame(all_keyes_list)
    """
    questions_df = pd.DataFrame(questions)
    questions_df[:5000].to_csv(os.path.join(test_path, "ндс_прибыль_5000.csv"))
    # questions_df.to_csv(os.path.join('./data', 'камеральная_проверка_вопросы.csv'))

    '''
    # запись дат запросов в JSON для того, чтобы сохранить информацию об уже обработанных запросах
    q_dates_dict = {'project': 'paraphrase_bss', 'dates': str(set(dts)), 'pub_ids': (6, 8, 9)}
    with open(os.path.join('./data', "date_quests.json"), 'w') as f:
        json.dump(q_dates_dict, f)
    '''
