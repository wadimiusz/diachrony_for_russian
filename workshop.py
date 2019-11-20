from algos import GlobalAnchors, Jaccard, KendallTau, ProcrustesAligner
from argparse import ArgumentParser
from utils import load_model
import os
from collections import OrderedDict
from tqdm import tqdm


def get_score(word, model1, model2):
    #return Jaccard(model1, model2, 1).get_score(word)
    #return ProcrustesAligner(model1, model2).get_score(word)
    return GlobalAnchors(model1, model2).get_score(word)

def main():
    parser = ArgumentParser()
    #parser.add_argument('--word', '-w', required=True, help='word to be scored')
    parser.add_argument('--model1', '-m1', required=True, help='Path to the word embeddings model')
    parser.add_argument('--model2', '-m2', required=True,
                        help='Path to the second word embedding model')
    parser.add_argument('--top-n-neighbors', '-n', dest='topn', default=50,
                        help='number of word neighbors to analyze '
                             '(optional, used in Kendall tau and Jaccard algo, default=50)')
    args = parser.parse_args()
    if not os.path.isfile(args.model1):
        raise FileNotFoundError("File {} not found".format(args.model1))

    if not os.path.isfile(args.model2):
        raise FileNotFoundError("File {} not found".format(args.model2))

    model1 = load_model(args.model1)
    model2 = load_model(args.model2)

    shared_words = list(set(model1.wv.vocab.keys()) & set(model2.wv.vocab.keys()))[:1000]
    print(len(shared_words))

    scores = [(w, get_score(w, model1, model2)) for w in tqdm(shared_words)]

    d = dict(scores)

    ordered = OrderedDict(sorted(d.items(), key=lambda t: -t[1]))

    with open('results_ga.txt', 'w') as f:
        for x in ordered.items():
            f.write('{0}:{1}'.format(x[0], x[1]))
            f.write('\n')


if __name__ == "__main__":
    main()





