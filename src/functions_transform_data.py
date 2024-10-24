import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# other
from collections import Counter
from gensim.models import Word2Vec

import numpy as np
import os
import random
import pandas as pd


# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from datasets import Dataset as datDataset

# other
from collections import Counter
from gensim.models import Word2Vec

from functions_helper import ConfigReader, KFoldReader, LabelMapper


# 0 ) creating vocab and optional word embeddings
class TrainVocabulary:
    def __init__(self, texts, unk_threshold, unk_token="[UNK]"):
        self.unk_token = unk_token
        self.unk_threshold = unk_threshold
        self.word2frq = Counter()
        for text in texts:
            self.word2frq += Counter(text)
        
        self.rare_words = dict(filter(lambda item: 0 < item[1] < unk_threshold, self.word2frq.items()))
        self.word2frq = dict(filter(lambda item: item[1] >= unk_threshold, self.word2frq.items()))

        self.word2idx = {self.unk_token : 1}
        self.word2idx.update({word : i for i, word in enumerate(self.word2frq.keys(), 2)})
        self.idx2word = {i : word for i, word in enumerate(self.word2idx.items())}
        self.word2frq[unk_token] = sum(self.rare_words.values())
        self.words = self.word2idx.keys()
        
    def __len__(self):
        assert len(self.word2idx) == len(self.idx2word) == len(self.word2frq)
        return len(self.word2idx)

    def create_word2vec(self, texts, vector_size=10, window=5, min_count=None):
        if min_count is None:
            min_count = self.unk_threshold
        texts_for_word_embeddings = texts.apply(self._unk_fn)
        self.word2vec = Word2Vec(sentences=texts_for_word_embeddings, vector_size=vector_size, window=window, min_count=min_count)
        assert len(self.words) == len(self.word2vec.wv), print(len(self.words), len(self.word2vec.wv))
        
        embedding_matrix = np.zeros(shape=(len(self.word2vec.wv)+1, vector_size), dtype=float)   # +1 for padding at pos 0
        for idx, word in enumerate(self.word2vec.wv.key_to_index.keys(), 1):    # start at 1 to keep 0 for padding
            embedding_matrix[idx] = self.word2vec.wv[word]
        assert embedding_matrix.shape == (len(self.word2vec.wv) +1, vector_size)    # +1 bc padding not in vocab
        return torch.tensor(embedding_matrix)
    
    def _unk_fn(self, text):
        return [token if token in self.words else self.unk_token for token in text]
    


# 1 ) Data preparation for Logistic Regression
class TF_IDF_Dataset:
    # create separate dataset for training (fit+transform), validation (transform), test (transform)
    def __init__(self, text_col, label_col, vocab, label_map, vectorizer_name, vectorizer, replace_oov=True):
        self.vocab = vocab
        self.label_col = label_col.reset_index(drop=True, inplace=False).map(label_map)
        self.vectorizer = None

        text_col = text_col.reset_index(drop=True, inplace=False)
        self.vectorizer_name = vectorizer_name
        if replace_oov:
            text_col = text_col.apply(self._unk_fn)
        if vectorizer is None:
            self.vectorized_matrix = self.fit_transform_tfidf(text_col)
        else:
            self.vectorized_matrix = self.transform_tfidf(text_col, vectorizer=vectorizer)
        self.n_features = self.vectorized_matrix.shape[1]
        
    def fit_transform_tfidf(self, tokenized_texts):
        if self.vectorizer_name == "tfidf":
            self.vectorizer = TfidfVectorizer(input="content", 
                                         tokenizer=self.dummy_tokenizer, 
                                         token_pattern=None, 
                                         lowercase=False,
                                         ngram_range=(1, 2)) # uni and bigrams
        elif self.vectorizer_name == "bow":
            self.vectorizer = CountVectorizer(input="content",
                                         tokenizer=self.dummy_tokenizer,
                                         token_pattern=None,
                                         lowercase=False)
        vectorized_docs = self.vectorizer.fit_transform(tokenized_texts)
        print(f"Fitted & transformed {self.vectorizer_name} matrix", vectorized_docs.shape)
        return vectorized_docs
    
    def transform_tfidf(self, tokenized_texts, vectorizer):
        vectorized_docs = vectorizer.transform(tokenized_texts)
        print(f"Transformed {self.vectorizer_name} matrix ", vectorized_docs.shape)
        return vectorized_docs

    def dummy_tokenizer(self, tokenized_text): 
        # returns the text as is, since it is pretokenized to save computing power
        return tokenized_text
    
    def _unk_fn(self, text):
        return [token if token in self.vocab.words else self.vocab.unk_token for token in text]
    
    def __getitem__(self, ind):
        # return self.X[index], self.y[index]
        return {"text": self.vectorized_matrix[ind].toarray().squeeze(), "label": self.label_col[ind]}
    
    def __len__(self):
        assert self.vectorized_matrix.shape[0] == len(self.label_col)
        return self.vectorized_matrix.shape[0]
    
    def collate_fn(self, batch, pad_value=0, batch_first=True):
    # batch = list of dicts {text, label}
        text_tensorlist = []
        labels = []
        for input_pair in batch:
            text_tensorlist.append(torch.tensor(input_pair["text"], dtype=torch.float32))
            labels.append(input_pair["label"])
        
        for tensor in text_tensorlist:
            assert tensor.shape[0] == self.n_features
        labels_tensor = torch.LongTensor(labels)
        # they are equal length already, so just packed, no padding is added
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        return padded_text_tensors, labels_tensor
    

# 2 ) Data Preparation for CNN (wordembeddings)
class CNNDataset(Dataset):  # add max doclen?
    def __init__(self, text_col, label_col, vocab, label_map, replace_oov=True, encode=True):
        self.vocab = vocab
        self.text_col = text_col.reset_index(drop=True, inplace=False)
        if replace_oov:
            self.text_col = self.text_col.apply(self._unk_fn)
        if encode:
            self.text_col = self.text_col.apply(self._encode_fn)
        
        self.label_col = label_col.reset_index(drop=True, inplace=False).map(label_map)

    def _unk_fn(self, text):
        return [token if token in self.vocab.words else self.vocab.unk_token for token in text]

    def _encode_fn(self, text):
        return [self.vocab.word2idx[token] for token in text]

    def __getitem__(self, ind):
        return {"text": self.text_col[ind], "label": self.label_col[ind]}
    
    def __len__(self):
        assert len(self.text_col) == len(self.label_col)
        return len(self.text_col)
    
    def collate_fn(self, batch, pad_value=0, batch_first=True):
    # batch = list of dicts {text, label}
        text_tensorlist = []
        labels = []
        for input_pair in batch:
            text_tensorlist.append(torch.LongTensor(input_pair["text"]))
            labels.append(input_pair["label"])

        labels_tensor = torch.LongTensor(labels)
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        return padded_text_tensors, labels_tensor

# 3 ) Dataset for BERT (with transformers library)
    
class BERTDataset:
    def __init__(self, df, textcol_name, labelcol_name, tokenizer, truncation=True, padding=False, batchsize=16):
        self.textcol_name= textcol_name
        self.labelcol_name = labelcol_name
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.padding = padding
        
        dataset = datDataset.from_pandas(df[[textcol_name, labelcol_name]], preserve_index=False)
        self.all_cols_dataset = dataset.map(self.encode_fn, batched=True, batch_size=batchsize)
        self.dataset = self.all_cols_dataset.remove_columns([textcol_name])
        if self.labelcol_name != "labels":
            self.dataset = self.dataset.rename_column(labelcol_name, "labels")
        self.dataset.set_format("pt", columns=self.dataset.column_names, output_all_columns=True)

        assert type(self.dataset["input_ids"][0]) == torch.Tensor
        assert type(self.dataset["labels"][0]) == torch.Tensor
        assert type(self.dataset["attention_mask"][0]) == torch.Tensor
        
        
    def encode_fn(self, sample):
        return self.tokenizer(sample[self.textcol_name], 
                                truncation=self.truncation, 
                                padding=self.padding, 
                                return_length=False)

    def collate_fn(self, batch, pad_value=0, batch_first=True):

        input_ids = [item["input_ids"] for item in batch]
        masks = [item["attention_mask"] for item in batch]
        labels = torch.LongTensor([item["labels"] for item in batch])

        padded_input_ids = pad_sequence(input_ids, batch_first=batch_first, padding_value=pad_value)
        padded_mask = pad_sequence(masks, batch_first=batch_first, padding_value=pad_value)
        return {"input_ids" : padded_input_ids,
                "attention_mask": padded_mask,
                "labels": labels}
    


# 4 ) HiSAN datset
    
class HiSANDataset(Dataset):
    def __init__(self, text_col, label_col, vocab, label_map, max_lines=256, max_words_per_line=20, replace_oov=True, encode=True):
        self.max_lines = max_lines
        self.max_words_per_line = max_words_per_line
        self.vocab = vocab
        self.text_col = text_col.reset_index(drop=True, inplace=False)
        if replace_oov:
            self.text_col = self.text_col.apply(self._unk_fn)
        if encode:
            self.text_col = self.text_col.apply(self._encode_fn)
        
        self.label_col = label_col.reset_index(drop=True, inplace=False).map(label_map)


    def _unk_fn(self, text):
        return [token if token in self.vocab.words else self.vocab.unk_token for token in text]

    def _encode_fn(self, text):
        return [self.vocab.word2idx[token] for token in text]

    def __getitem__(self, ind):
        return {"text": self.text_col[ind], "label": self.label_col[ind]}
    
    def __len__(self):
        assert len(self.text_col) == len(self.label_col)
        return len(self.text_col)
    
    def collate_fn(self, batch, pad_value=0, batch_first=True):
        # batch = list of dicts {text, label}
        text_tensorlist = []
        labels = []
        for input_pair in batch:
            text = input_pair["text"]
            max_padded_text = text + ((self.max_lines * self.max_words_per_line - len(text)) * [pad_value])
            assert len(max_padded_text) == self.max_lines*self.max_words_per_line
            text_tensorlist.append(torch.LongTensor(max_padded_text))
            labels.append(input_pair["label"])

        labels_tensor = torch.LongTensor(labels)
        # only packs sequence, it is already padded
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        return padded_text_tensors, labels_tensor
    

class HiSAN_SentDataset(Dataset):
    def __init__(self, text_col, label_col, vocab, label_map, max_lines=150, max_words_per_line=50, 
                 replace_oov=True, encode=True, pad_sents=True, padding_value=0):
        self.max_lines = max_lines
        self.max_words_per_line = max_words_per_line
        self.padding_value =padding_value
        self.vocab = vocab
        self.text_col = text_col.reset_index(drop=True, inplace=False)
        self.text_col = self.text_col.apply(self._truncate_left)
        if replace_oov:
            self.text_col = self.text_col.apply(self._unk_fn)
        if encode:
            self.text_col = self.text_col.apply(self._encode_fn)
        if pad_sents:
            self.text_col = self.text_col.apply(self._pad_sentences_to_maxlen)
        
        self.label_col = label_col.reset_index(drop=True, inplace=False).map(label_map)


    def _truncate_left(self, text):
        if len(text) > self.max_lines:
            assert len(text[-self.max_lines:]) == self.max_lines
            return text[-self.max_lines:]
        else:
            return text
    
    def _unk_fn(self, text):
        return [[token if token in self.vocab.words else self.vocab.unk_token for token in sentence] for sentence in text]

    def _pad_sentences_to_maxlen(self, text):
        padded_sents = []
        for sentence in text:
            padded_sent = sentence + (self.max_words_per_line - len(sentence)) * [self.padding_value]
            assert len(padded_sent) == self.max_words_per_line
            padded_sents.append(padded_sent)
        assert len(text) == len(padded_sents)   # same number of lists of lists
        return padded_sents
    
    def _encode_fn(self, text):
        return [[self.vocab.word2idx[token] for token in sentence] for sentence in text]

    def __getitem__(self, ind):
        return {"text": self.text_col[ind], "label": self.label_col[ind]}
    
    def __len__(self):
        assert len(self.text_col) == len(self.label_col)
        return len(self.text_col)
    
    def _flatten_list(self, nested_list):
        # to get flattened sents, but with sentence aware padding that is kept when slicing it to max_lines in model later on
        flattened = []
        for item in nested_list:
            flattened.extend(item)
        assert len(flattened) <= self.max_words_per_line * self.max_lines
        return flattened

    def collate_fn(self, batch, pad_value=0, batch_first=True):
        text_tensorlist = []
        labels = []
        for input_pair in batch:
            text = self._flatten_list(input_pair["text"])
            max_padded_text = text + ((self.max_lines * self.max_words_per_line - len(text)) * [pad_value])
            assert len(max_padded_text) == self.max_lines*self.max_words_per_line
            text_tensorlist.append(torch.LongTensor(max_padded_text))
            labels.append(input_pair["label"])

        labels_tensor = torch.LongTensor(labels)
        # only packs sequence, it is already padded
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        return padded_text_tensors, labels_tensor