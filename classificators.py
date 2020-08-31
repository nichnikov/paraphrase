# здесь будут объекты для создания правил
# https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
# https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1

import os, pickle, logging
from utility import Loader, AbstractRules, intersection, strings_similarities  # contrastive_loss
from texts_processors import TokenizerApply
from itertools import groupby
# from gensim.similarities import Similarity
from gensim.similarities import MatrixSimilarity
from statistics import mean


def include_and(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token not in text_list:
            return tuple((False, 0))
    return tuple((True, coeff))


def include_or(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token in text_list:
            return tuple((True, 0))
    return tuple((False, coeff))


def exclude_and(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token in text_list:
            return tuple((False, 0))
    return tuple((True, coeff))


def exclude_or(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token not in text_list:
            return tuple((True, coeff))
    return tuple((False, 0))


def include_str(tokens_str, text_str, coeff=0.0):
    if tokens_str in text_str:
        return tuple((True, coeff))
    else:
        return tuple((False, 0))


def exclude_str(tokens_str, text_str, coeff=0.0):
    if tokens_str not in text_str:
        return tuple((True, coeff))
    else:
        return tuple((False, 0))


def intersec_share(tokens_list, text_list, intersec_coeff=0.7):
    intersec_tks = intersection(tokens_list, text_list)
    intersec = len(intersec_tks) / len(tokens_list)
    if intersec > intersec_coeff:
        return tuple((True, intersec))
    else:
        return tuple((False, 0))


def include_str_p(tokens_list: list, txt_list: list, coeff):
    length = len(tokens_list)
    txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                  len(txt_list[i:i + length]) == length]
    for tx_l in txts_split:
        str_sim = strings_similarities(' '.join(tokens_list), ' '.join(tx_l))
        if str_sim >= coeff:  # self.sims_score:
            return tuple((True, str_sim))
    return tuple((False, 0))


def exclude_str_p(tokens_list: list, txt_list: list, coeff):
    length = len(tokens_list)
    txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                  len(txt_list[i:i + length]) == length]
    for tx_l in txts_split:
        if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
            return False
    return True


# функция, группирующая список кортежей [(1, a...), (2, b...), (1, c...)] по первому элементу в этих котрежах,
# возвращает список кортежей, каждый из которых имеет элементы предыдущего списка
# имеющие один тег [(1, [(a...), (c...)]), (2, [(b...)]))]
def model_params_grouped(model_params):
    return [(x, list(y)) for x, y in groupby(sorted(model_params, key=lambda x: x[0]),
                                             key=lambda x: x[0])]


class SimpleRules(AbstractRules):
    def __init__(self, loader_obj):
        self.functions_dict = {"include_and": include_and, "include_or": include_or,
                               "exclude_and": exclude_and, "exclude_or": exclude_or,
                               "include_str": include_str, "include_str_p": include_str_p,
                               "exclude_str_p": exclude_str_p, "intersec_share": intersec_share}
        self.model = loader_obj
        assert self.model.model_type == "simple_rules", "тип модели не соответствует классу SimpleRules"

        self.tokenizer = TokenizerApply(self.model)
        if not self.model.is_etalons_lemmatize:
            self.model.application_field["texts"] = self.tokenizer.texts_processing(self.model.application_field["texts"])

        self.model_params = list(zip(self.model.application_field["tags"],
                                     self.model.application_field["rules"],
                                     self.model.application_field["texts"],
                                     self.model.application_field["coeff"]))

        # grouping rules with the same tag
        self.model_params_grouped = model_params_grouped(self.model_params)

    # внешний метод:
    def rules_apply(self, texts, function_type="rules_apply_without_range"):
        if function_type == "rules_apply_without_range":
            return self.rules_apply_without_range(texts)

        elif function_type == "rules_apply_one":
            return self.rules_apply_one(texts)

        elif function_type == "rules_apply_range_one":
            return self.rules_apply_range_one(texts)

        elif function_type == "rules_apply_range":
            return self.rules_apply_range(texts)

        elif function_type == "rules_apply_range":
            return self.rules_apply_range(texts)

        elif function_type == "rules_apply_debugging":
            return self.rules_apply_debugging(texts)

    def rules_apply_without_range(self, texts):
        decisions = []
        model_params_group = model_params_grouped(self.model_params)
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for group, rules_list in model_params_group:
                decision = True
                for tg, rule, tknz_etalon, coeff in rules_list:
                    decision = decision and self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)[0]
                # будем возвращать только сработавшие правила (True)
                if decision:
                    decisions_temp.append(group)
            decisions.append((num, decisions_temp))
        return decisions

        # применение правил, когда мы точно знаем, что в эталонах одному правилу соответствует ровно одно условние

    def rules_apply_one(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for tg, rule, tknz_etalon, coeff in self.model_params:
                decision = self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)[0]
                # будем возвращать только сработавшие правила (True)
                if decision:
                    decisions_temp.append(decision)
            decisions.append((num, decisions_temp))
        return decisions

    def rules_apply_range_one(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            # оценка результатов применения правил для каждого тега (в каждой группе):
            decisions_temp = []
            for tg, rule, tknz_etalon, coeff in self.model_params:
                decision = self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                # будем возвращать только сработавшие правила (True)
                if decision[0]:
                    decisions_temp.append(tuple((tg, decision[1])))
            decisions.append((num, [tg for tg, scr in sorted(decisions_temp, key=lambda x: x[1], reverse=True)]))
        return decisions

    def rules_apply_range(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            # оценка результатов применения правил для каждого тега (в каждой группе):
            decisions_temp = []
            for group, rules_list in self.model_params_grouped:
                decision = True
                group_coeff = []
                for tg, rule, tknz_etalon, coeff in rules_list:
                    func_res = self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                    group_coeff.append(func_res[1])
                    decision = decision and func_res[0]
                # будем возвращать только сработавшие правила (True)
                if decision:
                    decisions_temp.append(tuple((group, mean(group_coeff))))
            decisions.append((num, [tg for tg, scr in sorted(decisions_temp, key=lambda x: x[1], reverse=True)]))
        return decisions

    def rules_apply_debugging(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        model_params_group = model_params_grouped(self.model_params)
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for group, rules_list in model_params_group:
                for tg, rule, tknz_etalon, coeff in rules_list:
                    decision = self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                    # будем возвращать только сработавшие правила (True)
                    decisions_temp.append((group, decision))
            decisions.append((num, decisions_temp))
        return decisions

class LsiClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model = loader_obj
        assert self.model.model_type == "lsi", "тип модели не соответствует классу SimpleRules"

        self.tknz = TokenizerApply(self.model)
        if 'index' in self.model.texts_algorithms:
            self.index = self.model.texts_algorithms['index']
        else:
            self.et_vectors = self.tknz.texts_processing(self.model.application_field["texts"])
            self.index = MatrixSimilarity(self.et_vectors, num_features=self.model.texts_algorithms["num_topics"])

        self.coeffs = self.model.application_field["coeff"]
        self.tags = self.model.application_field["tags"]

    # замечание: в эту функцию по хорошему надо добавить возможность нескольким правилам иметь разные эталоны
    # LSI модели, объединяемые через and
    def rules_apply(self, texts):
        text_vectors = self.tknz.texts_processing(texts)
        texts_tags_similarity = []
        for num, text_vector in enumerate(text_vectors):
            trues_list_scores = [(tg, scr, cf) for tg, scr, cf in list(zip(self.tags, self.index[text_vector],
                                                                           self.coeffs)) if scr > cf]
            # отсортируем, чтобы выводить наиболее подходящие результаты (с наибольшим скором)
            trues = [tg for tg, scr, cf in sorted(trues_list_scores, key=lambda x: x[1], reverse=True)]
            texts_tags_similarity.append((num, trues))
        return texts_tags_similarity



if __name__ == "__main__":
    import time

    models_path = r'./models'

    with open(os.path.join(models_path, "bss_model_lsi.pickle"), "br") as f:
        model_lsi = pickle.load(f)

    loader_obj = Loader(model_lsi)
    print(loader_obj.dictionaries)

    t1 = time.time()
    cl = LsiClassifier(Loader(model_lsi))
    print(time.time() - t1)

    tx = "упрощенная бухгалтерская отчетность кто сдает"
    t1 = time.time()
    rls = cl.rules_apply([tx])
    print(time.time() - t1)
    print(rls)

