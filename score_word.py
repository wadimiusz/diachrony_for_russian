from algos import GlobalAnchors, Jaccard, KendallTau, ProcrustesAligner

from argparse import ArgumentParser
from utils import load_model
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--word', '-w', required=True, help='word to be scored')
    parser.add_argument('--model1', '-m1', required=True, help='Path to the word embeddings model')
    parser.add_argument('--model2', '-m2', required=True, help='Path to the second word embedding model')
    parser.add_argument('--top-n-neighbors', '-n', dest='topn', default=50, help='number of word neighbors to analyze (optional, used in Kendall tau and Jaccard algo, default=50)')
    args = parser.parse_args()
    if not os.path.isfile(args.model1):
        raise FileNotFoundError("File {} not found".format(args.model1))

    if not os.path.isfile(args.model2):
        raise FileNotFoundError("File {} not found".format(args.model2))
    
    model1 = load_model(args.model1)
    model2 = load_model(args.model2)
    
    if args.word not in model1.vocab:
        raise ValueError("Word {} not in {}".format(args.word, args.model1))

    if args.word not in model2.vocab:
        raise ValueError("Word {} not in {}".format(args.word, args.model2))

    print("KendallTau score: {} (from -1 to 1)".format(KendallTau(model1, model2, top_n_neighbors=args.topn).get_score(args.word)))

    print("Jaccard score: {} (from 0 to 1)".format(Jaccard(model1, model2, args.topn).get_score(args.word)))

    print("Global Anchors score: {} (from -1 to 1)".format(GlobalAnchors(model1, model2).get_score(args.word)))

    print("Procrustes aligner score: {} (from -1 to 1)".format(ProcrustesAligner(model1, model2).get_score(args.word)))

if __name__ == "__main__":
    main()
