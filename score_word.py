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
    
    result = ProcrustesAligner(model1, model2).run_scores()
    
    for x, y in result[0:50]:
        m1_similars = [word for word, score in model1.most_similar(x, topn=10)]
        m2_similars = [word for word, score in model2.most_similar(x, topn=10)]           
        
        print(f"{x}: {y}")
        print("\tSoviet: " + ", ".join(m1_similars))
        print("\tPost-soviet: " + ", ".join(m2_similars))
        print()


if __name__ == "__main__":
    main()
