import os.path
from typing import Callable, Dict, Iterable, Optional, Union, List

import numpy as np
from textblob import TextBlob
from tqdm import tqdm

from .dataset import flatten


class Tokenizer:
    def __init__(self, tool: Callable):
        self.tool = tool
        self.corpus_tokens = None
        self.num_words = 0
        self.word_id: Dict[str, int] = dict()
        self.id_word: Dict[int, str] = dict()

    def fit(self,
            corpus: Iterable[Iterable[str]],
            stopwords: Iterable[str],
            spell_check=True,
            no_number=True):
        # size of corpus: n_corpus, n_texts
        result = []
        count = 0
        for texts in corpus:
            corpus_tokens = []
            for text in tqdm(texts, desc=f'corpus {count}'):
                if spell_check:
                    text = str(TextBlob(text).correct())
                tokens = []
                raw_words = self.tool(text)

                for word in raw_words:
                    if word not in stopwords:
                        if no_number and word.isnumeric():
                            continue
                        tokens.append(word)
                # prevent empty entries
                if not tokens:
                    tokens.append('none')
                corpus_tokens.append(tokens)
            result.append(corpus_tokens)

        # size of result: n_corpus, n_texts, n_words
        vocab = list(set(flatten(result)))
        self.num_words = len(vocab)
        id_mapper = [i for i in range(self.num_words)]
        self.word_id = dict(zip(vocab, id_mapper))
        self.id_word = dict(zip(id_mapper, vocab))
        self.corpus_tokens = result

        return result

    def encode(self, word_sequences: Iterable[Iterable[str]]):
        return [[self.word_id[word] for word in word_seq] for word_seq in word_sequences]

    def decode(self, id_sequences: Iterable[Iterable[int]]):
        return [[self.id_word[wid] for wid in id_seq] for id_seq in id_sequences]

    def fit_and_encode(self, corpus: Iterable[Iterable[str]], stopwords: Iterable[str]):
        corpus_tokens = self.fit(corpus, stopwords)

        return [[[self.word_id[word] for word in word_seq] for word_seq in word_sequences] for word_sequences in corpus_tokens]

    def save(self, dir_path: str, save_vocab=True):
        assert self.corpus_tokens, "Please fit a corpus first."

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(os.path.join(dir_path, 'corpus.txt'), mode='w') as f:
            for word_sequences in self.corpus_tokens:
                f.writelines([' '.join(words) + '\n' for words in word_sequences])
                f.write('\n---\n\n')

        if save_vocab:
            with open(os.path.join(dir_path, 'vocab.txt'), mode='w') as f:
                f.write('\t'.join(self.word_id.keys()))

    def load(self, dir_path: str):
        assert os.path.exists(dir_path), f"Cannot find directory {dir_path}."

        corpus_path = os.path.join(dir_path, 'corpus.txt')
        vocab_path = os.path.join(dir_path, 'vocab.txt')

        assert os.path.exists(corpus_path), f"No corpus found in directory {dir_path}."

        corpus_tokens = []
        with open(corpus_path) as f:
            word_sequences = []
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line != '':
                    if line == '---':
                        corpus_tokens.append(word_sequences)
                        word_sequences = []
                    else:
                        word_sequences.append(line.split())

        if os.path.exists(vocab_path):
            with open(vocab_path) as f:
                vocab = f.read().strip().split('\t')
        else:
            vocab = list(set(flatten(corpus_tokens)))

        self.num_words = len(vocab)
        id_mapper = list(range(self.num_words))
        self.word_id = dict(zip(vocab, id_mapper))
        self.id_word = dict(zip(id_mapper, vocab))

        return corpus_tokens


def load_weights(path: str,
                 vocab_id: Optional[Dict[str, int]] = None,
                 n_vocab: int = -1,
                 init_weights: Union[str, np.ndarray] = 'random',
                 embedding_dim: int = 100):
    """
    Load pre-trained word vectors from the given path.
    If vocab_id is given, the weight matrix would be initialized by init_weights.
    If not, it only loads n_vocab word vectors from the given path.

    :param path: the word vectors file path
    :param vocab_id: a dictionary with words as keys and integer id as values
    :param n_vocab: number of word vectors to load (only works when vocab_id is not given). -1 to load until EOF.
    :param init_weights: 'random' or 'zeros' or given a weight matrix (only works when vocab_id is given)
    :param embedding_dim: dimension of each word vector
    """

    vocab: List[str] = list()

    # initialize weight matrix
    if vocab_id:
        __weights_shape = (len(vocab_id), embedding_dim)
        if type(init_weights) == str:
            if init_weights == 'zeros':
                weights = np.zeros(shape=__weights_shape)
            else:
                weights = np.random.rand(*__weights_shape)
        else:
            weights = init_weights
    else:
        weights = None

    # load weights
    count = 0
    with open(path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                segments = line.split(' ')
                word = segments[0]
                word_vec = np.array(segments[1:])

                if vocab_id:
                    if word not in vocab_id:
                        continue

                vocab.append(word)
                if weights is None:
                    weights = word_vec
                elif weights.shape[0] == count:
                    weights = np.vstack((weights, word_vec))
                else:
                    weights[vocab_id[word]] = word_vec
                count += 1

                if weights is None and 0 <= n_vocab == count:
                    break
            else:
                break

    oov = list(set(vocab_id.keys()).difference(vocab))
    return vocab, oov, weights
