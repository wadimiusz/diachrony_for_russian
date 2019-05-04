import gensim
from scipy.stats import mstats
from utils import log


class KendallTau(object):
    def __init__(self, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                 top_n_neighbors):
        """
        :param w2v1: the model in question. if present, we use the index from that model
        :param w2v2: if word not present in w2v1, we look it up in the second model, w2v2
        """
        self.w2v1 = w2v1
        self.w2v2 = w2v2
        self.top_n_neighbors = top_n_neighbors

    def __repr__(self):
        return "KendallTau"

    def word_index(self, word: str) -> int:
        """
        A handy function for extracting the word index from models
        :param word: word the index of which we extract
        :return: the index of the word, an integer
        """
        if word in self.w2v1.wv:
            return self.w2v1.wv.vocab[word].index
        else:
            return len(self.w2v1.wv.vocab) + self.w2v2.wv.vocab[word].index

    def get_score(self, word: str):
        top_n_1 = [word for word, score in self.w2v1.most_similar(word, topn=self.top_n_neighbors)]
        top_n_2 = [word for word, score in self.w2v2.most_similar(word, topn=self.top_n_neighbors)]
        if len(top_n_1) == len(top_n_2) == self.top_n_neighbors:
            top_n_1 = [self.word_index(word) for word in top_n_1]
            top_n_2 = [self.word_index(word) for word in top_n_2]
            score, p_value = mstats.kendalltau(top_n_1, top_n_2)
            return score
        else:
            raise ValueError("Problem with word {word} and its neighbours".format(word=word))

    def get_changes(self, top_n_changed_words: int):
        log('Doing kendall tau')
        result = list()
        for num, word in enumerate(self.w2v1.wv.vocab.keys()):
            if num % 10 == 0:
                log("{words_num} / {length}".format(words_num=num, length=len(self.w2v1.wv.vocab)),
                    end='\r')

            score = self.get_score(word)
            result.append((word, score))

        result = sorted(result, key=lambda x: x[1])[:top_n_changed_words]
        log('\nDONE')
        return result


if __name__ == '__main__':
    pass
