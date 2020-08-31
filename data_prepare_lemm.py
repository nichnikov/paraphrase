import os, pickle
import pandas as pd
import random
from texts_processors import TokenizerApply
from utility import Loader

data_rout = r'./data'
models_rout = r'./models'

with open(os.path.join(models_rout, "tokenizator_model.pickle"), "br") as f:
    lingv_model = pickle.load(f)

tk_appl = TokenizerApply(Loader(lingv_model))
data_df = pd.read_csv(os.path.join(data_rout, "data_group_01.csv"))
lemm_txts_l = tk_appl.texts_processing(list(data_df['text']))
lemm_txts_df = pd.DataFrame(list(zip([" ".join(x) for x in lemm_txts_l], data_df['group'])))
lemm_txts_df.rename(columns={0: 'text', 1: 'group'}, inplace=True)
print(lemm_txts_df)

lemm_txts_df.to_csv(os.path.join(data_rout, "lemm_data_group_01.csv"), index=False, columns=['text', 'group'])
df = pd.read_csv(os.path.join(data_rout, "lemm_data_group_01.csv"))
print(df)


# герерация пар семантически одинаковых вопросов
lbs = set(df['group'])
results_tuples = []
for lb in lbs:
    work_list = list(df['text'][df['group'] == lb])
    for tx1 in work_list:
        for tx2 in work_list:
            results_tuples.append((tx1, tx2, 1))

# генерация пар семантически не эквивалентных вопросов:
result_df = pd.DataFrame(results_tuples, columns=["context", "response", "label"])

lbs_df = pd.DataFrame(lbs, columns=['group'])


# нужно сформировать все возможные пары групп вопросов:
# algorithm

# из этих групп сформировать обучающуюся выборку
print(lbs)
print(len(lbs))

results_tuples_diff_lbs = []
for lb in lbs:
    txts_list_1 = list(df['text'][df['group'] == lb])
    txts_list_2 = list(df['text'][df['group'] != lb])
    for tx1 in txts_list_1:
        for tx2 in txts_list_2:
            results_tuples_diff_lbs.append((tx1, tx2, 0))

result_pair_df = pd.DataFrame(results_tuples, columns=["question1", "question2", "is_duplicate"])
result_no_pairs_df = pd.DataFrame(results_tuples_diff_lbs, columns=["question1", "question2", "is_duplicate"])

print(result_pair_df)
print(result_no_pairs_df)

result_pair_df.to_csv(os.path.join(data_rout, 'b_paraphrase.csv'))
result_no_pairs_df.to_csv(os.path.join(data_rout, 'b_no_paraphrase.csv'))