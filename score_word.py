from algos import GlobalAnchors, Jaccard, KendallTau, ProcrustesAligner
from argparse import ArgumentParser
from utils import load_model
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--word', '-w', required=True, help='word to be scored')
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

    # print("Procrustes aligner score: {} (from -1 to 1)".format(
    
    xyz0 = ProcrustesAligner(model1, model2).run_scores()
    
    for x, y in xyz0[0:500]:
        print(f"{x}: {y}")


if __name__ == "__main__":
    main()
