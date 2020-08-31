import os, pickle, logging, keras
from utility import AbstractRules, ModelsChain
from texts_processors import TokenizerApply
import tensorflow as tf
import numpy as np
from keras import backend as K


def contrastive_loss(y_true, y_pred):
    # ontrastive loss from Hadsell-et-al.'06
    # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf'''

    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


keras.losses.contrastive_loss = contrastive_loss


class SiameseNnDoc2VecClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = "siamese_lstm_d2v"
        self.model = loader_obj
        self.tknz = TokenizerApply(self.model)
        self.tkz_model = self.tknz.model_tokenize()

    def rules_apply(self, texts):
        text_vectors = self.tknz.texts_processing(texts)
        et_vectors = self.tkz_model.application_field["texts"]
        coeffs = self.tkz_model.application_field["coeff"]
        tags = self.tkz_model.application_field["tags"]

        decisions = []
        vcs_arr = np.array(et_vectors)

        global graph
        graph = tf.get_default_graph()

        for num, text_vector in enumerate(text_vectors):
            tx_tensor = np.array([text_vector for i in range(vcs_arr.shape[0])])
            tx_tensor = tx_tensor.reshape(vcs_arr.shape[0], vcs_arr.shape[1], 1)
            vcs_arr = vcs_arr.reshape(vcs_arr.shape[0], vcs_arr.shape[1], 1)
            with graph.as_default():
                scores = self.model.classificator_algorithms["siamese_lstm_model"].predict([tx_tensor, vcs_arr])
            trues = [(tg, True) for scr, cf, tg in zip(scores, coeffs, tags) if scr < cf]
            falses = [(tg, False) for scr, cf, tg in zip(scores, coeffs, tags) if scr > cf]
            decisions.append((num, trues + falses))

        return decisions


if __name__ == "__main__":
    import time
    from utility import Loader

    data_rout = r'./data'
    models_rout = r'./models'

    with open(os.path.join(models_rout, "fast_answrs", "bss_siamese_lstm_d2v.pickle"), "br") as f:
        bss_siamese = pickle.load(f)

    tx = ["кто может применять упрощенный баланс"]
    mdschain = ModelsChain([Loader(bss_siamese)], classes=[SiameseNnDoc2VecClassifier])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_siamese:", rt_t, time.time() - t1)