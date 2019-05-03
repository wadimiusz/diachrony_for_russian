import gensim
import numpy as np
from utils import log
from utils import intersection_align_gensim


def smart_procrustes_align_gensim(base_embed: gensim.models.KeyedVectors,
                                  other_embed: gensim.models.KeyedVectors):
    """
    This code, taken from
    https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and modified,
    uses procrustes analysis to make two word embeddings compatible.
    :param base_embed: first embedding
    :param other_embed: second embedding to be changed
    :return other_embed: changed embedding
    """
    base_embed.init_sims()
    other_embed.init_sims()

    base_vecs = base_embed.syn0norm
    other_vecs = other_embed.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = other_embed.syn0norm.dot(ortho)

    return other_embed


class ProcrustesAligner(object):
    def __init__(self, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors):
        self.w2v1, self.w2v2 = intersection_align_gensim(w2v1, w2v2)
        self.w2v2_changed = smart_procrustes_align_gensim(w2v1, w2v2)

    def __repr__(self):
        return "ProcrustesAligner"

    def get_score(self, word):
        vector1 = self.w2v1.wv[word]
        vector2 = self.w2v2_changed.wv[word]
        # score = cosine_similarity(vector1.reshape((1, -1)), vector2.reshape((1, -1)))[0][0]
        score = np.dot(vector1, vector2)  # More straightforward computation
        return score

    def get_changes(self, top_n_changed_words: int):
        log('Doing procrustes')
        result = list()
        # their vocabs should be the same, so it doesn't matter over which to iterate:
        for word in self.w2v1.wv.vocab.keys():
            score = self.get_score(word)
            result.append((word, score))

        result = sorted(result, key=lambda x: x[1])
        result = result[:top_n_changed_words]
        log('Done')
        return result


if __name__ == "__main__":
    pass
