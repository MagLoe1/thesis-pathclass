import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# 1) Data preparation for KBBERT dataset

class MTBERTDataset(Dataset):
    def __init__(self, df_split, text_col, label_cols, 
                 task_manager, tokenizer, 
                 truncation=True, padding=False, encode=True, batchsize=16):
        self.tokenizer = tokenizer
        self.truncation =truncation
        self.padding = padding

        self.label_cols = label_cols
        self.df_text_col = df_split[text_col].reset_index(drop=True, inplace=False)

        if encode:
            self.df_text_col = self.df_text_col.apply(self._encode_fn)


        self.df_label_cols = pd.DataFrame()
        self.label_cols = sorted(label_cols)    # ensure it is sorted
        for label_col in self.label_cols:
            labels = df_split[label_col].reset_index(drop=True, inplace=False).map(task_manager[label_col]["label_map"])
            self.df_label_cols[label_col] = labels
    
    def _encode_fn(self, text):
        return self.tokenizer(text, 
                            truncation=self.truncation, 
                            padding=self.padding, 
                            return_length=False)
    
    def __len__(self):
        assert len(self.df_text_col) == len(self.df_label_cols)
        return len(self.df_text_col)
    
    def __getitem__(self, ind):
        output = self.df_text_col.loc[ind]
        for label_col in self.label_cols:
            output[label_col] = self.df_label_cols[label_col][ind]
        return output

    def _collate_fn(self, batch, pad_value=0, batch_first=True):
        input_ids_list = []
        attention_masks_list = []
        labels = defaultdict(list)
        for inputs in batch:
            input_ids_list.append(torch.LongTensor(inputs["input_ids"]))
            attention_masks_list.append(torch.LongTensor(inputs["attention_mask"]))
            for task in self.label_cols:
                labels[task].append(inputs[task])

        for task in labels.keys():
            labels[task] = torch.LongTensor(np.array(labels[task]))
        padded_input_ids = pad_sequence(input_ids_list, batch_first=batch_first, padding_value=pad_value)
        padded_masks = pad_sequence(attention_masks_list, batch_first=batch_first, padding_value=pad_value)
        input_ids = {"input_ids": padded_input_ids}
        attention_masks = {"attention_mask" : padded_masks}
        return {**input_ids, **attention_masks, **labels}


# 2 ) Data Preparation for MTCNN (word embeddings)
class MTCNN_Dataset(Dataset):
    def __init__(self, df_split, text_col, label_cols, vocab, task_manager, replace_oov=True, encode=True):
        self.label_cols = label_cols
        self.vocab = vocab
        self.df_text_col = df_split[text_col].reset_index(drop=True, inplace=False)
        if replace_oov:
            self.df_text_col = self.df_text_col.apply(self._unk_fn)
        if encode:
            self.df_text_col = self.df_text_col.apply(self._encode_fn)

        # create dataframe
        self.df_label_cols = pd.DataFrame()
        self.label_cols = sorted(label_cols)    # ensure it is sorted
        for label_col in self.label_cols:
            labels = df_split[label_col].reset_index(drop=True, inplace=False).map(task_manager[label_col]["label_map"])
            self.df_label_cols[label_col] = labels
        
    def _unk_fn(self, text):
        return [token if token in self.vocab.words else self.vocab.unk_token for token in text]

    def _encode_fn(self, text):
        return [self.vocab.word2idx[token] for token in text]

    def __getitem__(self, ind):
        output = {"text": self.df_text_col.loc[ind]}
        for label_col in self.label_cols:
            output[label_col] = self.df_label_cols[label_col][ind]
        return output
    
    def __len__(self):
        assert len(self.df_text_col) == len(self.df_label_cols)
        return len(self.df_text_col)
    
    def collate_fn(self, batch, pad_value=0, batch_first=True):
    # batch = list of dicts {text, task1, task2}
        text_tensorlist = []
        labels = defaultdict(list)
        for inputs in batch:
            text_tensorlist.append(torch.LongTensor(inputs["text"]))
            for task in self.label_cols:
                labels[task].append(inputs[task])

        for task in labels.keys():
            labels[task] = torch.LongTensor(np.array(labels[task]))
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        texts = {"text": padded_text_tensors}
        return {**texts, **labels}
    
## Data prep for MT Hisan Sentence based
class MTHiSAN_SentDataset(Dataset):

    def __init__(self, df_split, text_col, label_cols, vocab, task_manager, max_lines=150, max_words_per_line=50, 
                 replace_oov=True, encode=True, pad_sents=True, padding_value=0):
        self.max_lines = max_lines
        self.max_words_per_line = max_words_per_line
        self.padding_value =padding_value
        self.vocab = vocab

        self.df_text_col = df_split[text_col].reset_index(drop=True, inplace=False)
        self.df_text_col = self.df_text_col.apply(self._truncate_left)
        if replace_oov:
            self.df_text_col = self.df_text_col.apply(self._unk_fn)
        if encode:
            self.df_text_col = self.df_text_col.apply(self._encode_fn)
        if pad_sents:
            self.df_text_col = self.df_text_col.apply(self._pad_sentences_to_maxlen)
        
        self.df_label_cols = pd.DataFrame()
        self.label_cols = sorted(label_cols)
        for label_col in self.label_cols:
            labels = df_split[label_col].reset_index(drop=True, inplace=False).map(task_manager[label_col]["label_map"])
            self.df_label_cols[label_col] = labels


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
        output = {"text": self.df_text_col.loc[ind]}
        for label_col in self.label_cols:
            output[label_col] = self.df_label_cols[label_col][ind]
        return output
    
    def __len__(self):
        assert len(self.df_text_col) == len(self.df_label_cols)
        return len(self.df_text_col)
    
    def _flatten_list(self, nested_list):
        # to get flattened sents, but with sentence aware padding that is kept when slicing it to max_lines in model later on
        flattened = []
        for item in nested_list:
            flattened.extend(item)
        assert len(flattened) <= self.max_words_per_line * self.max_lines
        return flattened

    def collate_fn(self, batch, pad_value=0, batch_first=True):
        # batch = list of dicts {text, label}
        text_tensorlist = []
        labels = defaultdict(list)
        for inputs in batch:
            text = self._flatten_list(inputs["text"])
            max_padded_text = text + ((self.max_lines * self.max_words_per_line - len(text)) * [pad_value])
            assert len(max_padded_text) == self.max_lines*self.max_words_per_line
            text_tensorlist.append(torch.LongTensor(max_padded_text))
            for task in self.label_cols:
                labels[task].append(inputs[task])
        for task in labels.keys():
            labels[task] = torch.LongTensor(np.array(labels[task]))

        # only packs sequence, it is already padded
        padded_text_tensors = pad_sequence(text_tensorlist, padding_value=pad_value, batch_first=batch_first)
        texts = {"text" : padded_text_tensors}
        return {**texts, **labels}
