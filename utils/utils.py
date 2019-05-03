import os
import sys
import gensim
import numpy as np
import logging
import functools


def informative_output(words_and_scores, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                       top_n_neighbors: int, model_name: str):
    print(model_name.center(40, '='))

    for word, score in words_and_scores:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        print("word {word} has score {score}".format(word=word, score=score))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*top_n_1, sep=',')
        print('_' * 40)
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*top_n_2, sep=',')
        print("")


def simple_output(words_and_scores, model_name):
    print(model_name.center(40, '='))
    print(*[word for word, score in words_and_scores], sep='\n')
    print('')


def log(message: str, end: str = '\n'):
    """Used for logging. use 2> /dev/null to swich off log messages"""
    sys.stderr.write(message+end)
    sys.stderr.flush()


def format_time(time: float):
    hours = int(time // 3600)
    minutes = int(time % 3600) // 60
    seconds = time % 60
    return "{h}h {m}m {s:.2f}s".format(h=hours,m=minutes,s=seconds)


def load_model(embeddings_file):
    """
    This function, written by github.com/akutuzov, unifies various standards of word embedding files.
    It automatically determines the format by the file extension and loads it from the disk correspondingly.
    :param embeddings_file: path to the file
    :return: the loaded model
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError("No file called {file}".format(file=embeddings_file))
    # Determine the model format by the file extension
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Binary word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Text word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)

    return emb_model


def intersection_align_gensim(m1: gensim.models.KeyedVectors, m2: gensim.models.KeyedVectors,
                              pos_tag: (str, None) = None, words: (list, None) = None):
    """
    This procedure, taken from https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and slightly
    modified, corrects two models in a way that only the shared words of the vocabulary are kept in the model,
    and both vocabularies are sorted by frequencies.
    Original comment is as follows:

    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new vectors and vectors_norm objects in both gensim models:
        -- so that Row 0 of m1.vectors will be for the same word as Row 0 of m2.vectors
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.

    :param m1: the first model
    :param m2: the second model
    :param pos_tag: if given, we remove words with other pos tags
    :param words: a container
    :return m1, m2: both models after their vocabs are modified
    """

    # Get the vocab for each model
    if pos_tag is None:
        vocab_m1 = set(m1.vocab.keys())
        vocab_m2 = set(m2.vocab.keys())
    else:
        vocab_m1 = set(word for word in m1.vocab.keys() if word.endswith("_" + pos_tag))
        vocab_m2 = set(word for word in m2.vocab.keys() if word.endswith("_" + pos_tag))

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return m1, m2

    # Otherwise sort lexicographically
    common_vocab = list(common_vocab)
    common_vocab.sort()

    # Then for each model...
    for m in (m1, m2):
        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.vectors_norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors_norm = m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = dict()
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab

    return m1, m2
