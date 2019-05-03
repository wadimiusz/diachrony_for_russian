import gensim
import numpy as np
from utils import log, intersection_align_gensim
from gensim.matutils import unitvec


class GlobalAnchors(object):
    def __init__(self, w2v1, w2v2, assume_vocabs_are_identical=False):
        if not assume_vocabs_are_identical:
            w2v1, w2v2 = intersection_align_gensim(w2v1, w2v2)
        
        self.w2v1 = w2v1
        self.w2v2 = w2v2

    def __repr__(self):
        return "GlobalAnchors"

    def get_global_anchors(self, word: str, w2v: gensim.models.KeyedVectors):
        """
        This takes in a word and a KeyedVectors model and returns a vector of cosine distances
        between this word and each word in the vocab.
        :param word:
        :param w2v:
        :return: np.array of distances shaped (len(w2v.vocab),)
        """
        word_vector = w2v.get_vector(word)
        similarities = gensim.models.KeyedVectors.cosine_similarities(word_vector, w2v.vectors)
        return unitvec(similarities)

    def get_score(self, word: str):
        w2v1_anchors = self.get_global_anchors(word, self.w2v1)
        w2v2_anchors = self.get_global_anchors(word, self.w2v2)

        # score = gensim.models.KeyedVectors.cosine_similarities(
        # w2v1_anchors, w2v2_anchors.reshape(1, -1))[0]
        score = np.dot(w2v1_anchors, w2v2_anchors)
        return score

    def get_changes(self, top_n_changed_words: int):
        """
        This method uses approach described in
        Yin, Zi, Vin Sachidananda, and Balaji Prabhakar.
        "The global anchor method for quantifying linguistic shifts and
        domain adaptation." Advances in Neural Information Processing Systems. 2018.
        It can be described as follows. To evaluate how much the meaning of a given word differs in
        two given corpora, we take cosine distance from the given word to all words
        in the vocabulary; those values make up a vector with as many components as there are words
        in the vocab. We do it for both corpora and then compute the cosine distance
        between those two vectors
        :param top_n_changed_words: we will output n words that differ the most in the given corpora
        :return: list of pairs (word, score), where score indicates how much a word has changed
        """
        log('Doing global anchors')
        result = list()
        for num, word in enumerate(self.w2v1.wv.vocab.keys()):
            if num % 10 == 0:
                log("{num} / {length}".format(num=num, length=len(self.w2v1.wv.vocab)), end='\r')

            score = self.get_score(word)
            result.append((word, score))

        result = sorted(result, key=lambda x: x[1])
        result = result[:top_n_changed_words]
        log('\nDone')
        return result
