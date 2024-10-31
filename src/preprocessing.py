from collections import Counter
from gensim.models import Word2Vec
import torch
import numpy as np
from string import punctuation



# unk:1, to keep 0 idx for fixed padding idx
class Vocabulary:
    def __init__(self, texts, set_unk=False, unk_token="[UNK]", unk_threshold=1):
        # if type(texts[0]) == str:
        #     print("Not a list", texts[0])
        self.unk_token = unk_token
        self.unk_threshold = unk_threshold
        self.word2frq = Counter()
        for text in texts:
            self.word2frq += Counter(text)
        
        self.word2idx = {self.unk_token : 1}
        self.word2idx.update({word : i for i, word in enumerate(self.word2frq.keys(), 2)})
        self.idx2word = {i : word for i, word in enumerate(self.word2idx.items())}

        self.words = self.word2idx.keys()

        self.numbers, self.punctuation, self.single_letter = self._get_subgroups()
        
    def __len__(self):
        assert len(self.word2idx) == len(self.idx2word) == len(self.word2frq)
        return len(self.word2idx)
    
    def _get_subgroups(self):
        numbers = dict()
        punct = dict()
        single_letter = dict()

        for token , count in self.word2frq.items():
            if token.isnumeric():
                numbers[token] = count
            if len(token) == 1:
                single_letter[token] = count
            if token in punctuation:
                punct[token] = count
        return numbers, punct, single_letter
        
    def vocabsize(self, min_count):
        return len(dict(filter(lambda item: 0 < item[1] >= min_count, self.word2frq.items())))
        

    def _unk_fn(self, text):
            return [token if token in self.words else self.unk_token for token in text]

    def create_word2vec(self, texts, vector_size=100, window=5, min_count=None):
        if min_count is None:
            min_count = self.unk_threshold
        texts_for_word_embeddings = texts.apply(self._unk_fn)
        self.word2vec = Word2Vec(sentences=texts_for_word_embeddings, vector_size=vector_size, window=window, min_count=min_count)
        assert len(self.words) == len(self.word2vec.wv), print(len(self.words), len(self.word2vec.wv))
        
        embedding_matrix = np.zeros(shape=(len(self.word2vec.wv)+1, vector_size), dtype=float)   # +1 for padding at pos 0
        for idx, word in enumerate(self.word2vec.wv.key_to_index.keys(), 1):    # start at 1 to keep 0 for padding
            embedding_matrix[idx] = self.word2vec.wv[word]
        assert embedding_matrix.shape == (len(self.word2vec.wv) +1, vector_size)    # +1 since padding not in vocab
        return torch.tensor(embedding_matrix)
    
 