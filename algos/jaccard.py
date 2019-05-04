import gensim
from utils import log


class Jaccard(object):
    def __init__(self, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                 top_n_neighbors: int):

        self.w2v1 = w2v1
        self.w2v2 = w2v2
        self.top_n_neighbors = top_n_neighbors

    def __repr__(self):
        return "Jaccard"

    def get_score(self, word: str):
        top_n_1 = [word for word, score in self.w2v1.most_similar(word, topn=self.top_n_neighbors)]
        top_n_2 = [word for word, score in self.w2v2.most_similar(word, topn=self.top_n_neighbors)]
        if len(top_n_1) == self.top_n_neighbors and len(top_n_2) == self.top_n_neighbors:
            intersection = set(top_n_1).intersection(set(top_n_2))
            union = set(top_n_1 + top_n_2)
            score = len(intersection) / len(union)
            return score
        else:
            raise ValueError("Problem with {word} and its neighbours".format(word=word))

    def get_changes(self, top_n_changed_words: int):
        log('Doing jaccard')
        result = list()
        for num, word in enumerate(self.w2v1.wv.vocab):
            if num % 10 == 0:
                log("{words_num} / {length}".format(
                    words_num=num, length=len(self.w2v1.wv.vocab.keys())), end='\r')

            score = self.get_score(word=word)
            result.append((word, score))

        result = sorted(result, key=lambda x: x[1])[:top_n_changed_words]
        log('\nDone')
        return result


if __name__ == '__main__':
    pass
