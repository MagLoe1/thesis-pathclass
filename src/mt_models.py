# imports

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from transformers import AutoModel



# CNN
class MTCNN(nn.Module):
    def __init__(self, 
                emb_matrix,
                embedding_dim,
                n_filters_per_size,
                filter_sizes, # for now: hard coded for exactly 3 sizes (e.g.for 3,4,5-grams)
                task_dict,  
                device,
                dropout=0.5,
                padding_value=0):
    
        super(MTCNN, self).__init__()
        self.name = f"MT_CNN"
        self.is_transformer = False
        self.task_dict = task_dict
        self.n_classes = {task: value["n_classes"] for task, value in self.task_dict.items()}
        self.tasks = list(task_dict.keys())
        emb_matrix.to(device)
        self.embedding = nn.Embedding(emb_matrix.shape[0], emb_matrix.shape[1], 
                                      padding_idx=padding_value, 
                                      dtype=torch.float32).from_pretrained(emb_matrix, freeze=False)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters_per_size, kernel_size=filter_sizes[0], padding=0),
                                   nn.ReLU(),
                                   nn.AdaptiveMaxPool1d(1))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters_per_size, kernel_size=filter_sizes[1], padding=0),
                                   nn.ReLU(),
                                   nn.AdaptiveMaxPool1d(1))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters_per_size, kernel_size=filter_sizes[2], padding=0),
                                   nn.ReLU(),
                                   nn.AdaptiveMaxPool1d(1))
        
        
        self.dropout = nn.Dropout(dropout)

        self.fcs = nn.ModuleDict()
        for task, task_values in task_dict.items():
            fc = nn.Linear(in_features=(n_filters_per_size * len(filter_sizes)), out_features=task_values["n_classes"])
            self.fcs[task] = fc
        self.softmax = nn.LogSoftmax(dim=1) # Logsoftmax for NLLLoss later on


    def forward(self, x):

        fw_outputs = {}

        x = self.embedding(x)
        # to get shape (batchsize, emb_dim, seq_length)
        x = torch.transpose(x, 1, 2).type(torch.float32)    
        out_conv1 = self.conv1(x)    # shape: batchsize, n_filters_per_size, 1
        out_conv2 = self.conv2(x)    # shape: batchsize, n_filters_per_size, 1
        out_conv3 = self.conv3(x)    # shape: batchsize, n_filters_per_size, 1

        # concatenate all filters to shape [batchsize, n_filters_per_size*n_filtersizes, 1]
        feature_map = torch.cat([out_conv1, out_conv2, out_conv3], dim=1)   
        feature_map = torch.flatten(feature_map, start_dim=1)
        feature_map = self.dropout(feature_map)
        
        for task, fc in self.fcs.items():
            logits = fc(feature_map)
            out_probs = self.softmax(logits)
            fw_outputs[task] = out_probs
        return fw_outputs
    



# 3 BERT
class MTKB_LongBERT(nn.Module):
    def __init__(self,
                 bert_model_path,
                 task_dict,
                 tokenizer,
                 device,
                 freeze_bert="partial",
                 dropout=0.1,
                 chunksize=512):
        super(MTKB_LongBERT, self).__init__()
        self.task_dict = task_dict
        self.n_classes = {task: value["n_classes"] for task, value in self.task_dict.items()}
        self.tasks = list(task_dict.keys())
        
        self.tokenizer = tokenizer
        self.device = device
        self.chunksize = chunksize
        self.kb_bert_layer = AutoModel.from_pretrained(bert_model_path, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        
        self.fcs = nn.ModuleDict()
        for task, task_values in task_dict.items():
            fc = nn.Linear(in_features=768, out_features=task_values["n_classes"])
            self.fcs[task] = fc

        self.softmax = nn.LogSoftmax(dim=1)
        self.is_transformer = True

        # be sure that embeddings have all added tokens
        self.kb_bert_layer.resize_token_embeddings(len(self.tokenizer)) # to account for any new tokens added to the tokenizer

        # trainable classification layers
        for fc_name, fc in self.fcs.items():
            for params in fc.parameters():
                params.requires_grad = True

        if not freeze_bert:
            print("Error: Some parameters need to be frozen. Change config file.")
            exit()
        
        # freeze all bert layers
        for params in self.kb_bert_layer.parameters():
            params.requires_grad = False

        # only freeze some layers:
        if freeze_bert == "partial" or freeze_bert == "partial-pool":

            # unfreeze last two      
            for name, params in self.kb_bert_layer.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name:
                    params.requires_grad = True
        # unfreeze the pooler output layer
        if freeze_bert == "all-pool" or freeze_bert == "partial-pool":
            for name, params in self.kb_bert_layer.named_parameters():
                if "pooler" in name:
                    params.requires_grad = True


    def forward(self, input_ids, mask):
        
        fw_outputs = {}
        # if whole batch has texts with len(text) < 512 tokens: return directly (faster)
        if input_ids.shape[1] <= self.chunksize:
            bert_out = self.kb_bert_layer(input_ids, mask, return_dict=True)

            cls_token_out = bert_out["pooler_output"]  # (batchsize, sequence length, embedding_dim) -> take the embedding of the CLS (== 0th) token of the sequence for each batch
            cls_token_out = self.dropout(cls_token_out)


            for task, fc in self.fcs.items():
                logits = fc(cls_token_out)
                out_probs = self.softmax(logits)
                fw_outputs[task] = out_probs
            return fw_outputs
        
        ###########
        
        ## else: use maxpooling as described in Gao et al. (2021) to handle longer sequences:
        ## classify chunks, then classify final
        collecting_logits = dict()
        for task, values in self.task_dict.items():
            collecting_logits[task] = torch.zeros(input_ids.shape[0], values["n_classes"]).to(self.device) # collect logits for each text -> feed it through softmax as a batch tensor again


        max_raw_input = self.chunksize-2

        for i, input_ids_one_text in enumerate(input_ids):
            # if text is short, pass directly through and get logits for this text
            if sum(mask[i]) <= self.chunksize:

                bert_out = self.kb_bert_layer(input_ids_one_text[:self.chunksize].unsqueeze(0), mask[i][:self.chunksize].unsqueeze(0), return_dict=True)
                # -> take the embedding of the CLS (== 0th) token of the sequence for each batch
                cls_token_out = bert_out["pooler_output"]
                cls_token_out = self.dropout(cls_token_out)
                for task, fc in self.fcs.items():
                    logits = fc(cls_token_out)
                    collecting_logits[task][i] = logits
                

            # else: split long text into k chunks, classify (like a batch) and 
            # merge k logits into a single logits with maxpooling
            else: 
                unpadded_input_ids = input_ids_one_text[1:sum(mask[i])-1]  # exclude [SEP] token at the end
                
                assert self.tokenizer.sep_token_id not in unpadded_input_ids
                assert self.tokenizer.cls_token_id not in unpadded_input_ids
                split_chunks = torch.split(unpadded_input_ids, split_size_or_sections=max_raw_input)
                padded_chunks = list()
                padded_chunks_mask = list()
                for chunk in split_chunks:

                    chunk = torch.cat((torch.tensor([self.tokenizer.cls_token_id]).to(self.device),
                                        chunk,
                                        torch.tensor([self.tokenizer.sep_token_id]).to(self.device)))
                    
                    chunk_mask = torch.ones_like(chunk)

                    if len(chunk) < self.chunksize:
                        n_padding = self.chunksize - len(chunk)
                        chunk = torch.nn.functional.pad(chunk, pad=(0, n_padding), value=self.tokenizer.pad_token_id)
                        chunk_mask = torch.nn.functional.pad(chunk_mask, pad=(0, n_padding), value=self.tokenizer.pad_token_id)

                    padded_chunks.append(chunk)
                    padded_chunks_mask.append(chunk_mask)
                    
     
                all_chunks = torch.vstack(padded_chunks).to(self.device)
                all_chunks_mask = torch.vstack(padded_chunks_mask).to(self.device)
    
                assert all_chunks.shape[1] == self.chunksize
 
                bert_out = self.kb_bert_layer(all_chunks, all_chunks_mask, return_dict=True)
                 # -> take the embedding of the CLS (== 0th) token of the sequence for each batch
                cls_token_out = bert_out["pooler_output"]  # (n_chunks, sequence length, embedding_dim)
                cls_token_out = self.dropout(cls_token_out)

                for task, fc in self.fcs.items():
                    logits = fc(cls_token_out)
                    collecting_logits[task][i] = torch.max(logits, dim=0).values # dim to collapse = n_chunks
        
        
        for task, fc in self.fcs.items():
            out_probs = self.softmax(collecting_logits[task])
            fw_outputs[task] = out_probs
        return fw_outputs
    


# MT-HISAN by (Gao et al.) with small adaptations

class MTHiSAN(nn.Module):
    '''
    Multitask hierarchical self-attention network for classifying cancer pathology reports.

    Args:
        embedding_matrix (numpy array): Numpy array of word embeddings.
            Each row should represent a word embedding.
            NOTE: The word index 0 is masked, so the first row is ignored.
        num_classes (list[int]): Number of possible output classes for each task.
        max_words_per_line (int): Number of words per line.
            Used to split documents into smaller chunks.
        max_lines (int): Maximum number of lines per document.
            Additional lines beyond this limit are ignored.
        att_dim_per_head (int, default: 50): Dimension size of output from each attention head.
            Total output dimension is att_dim_per_head * att_heads.
        att_heads (int, default: 8): Number of attention heads for multihead attention.
        att_dropout (float, default: 0.1): Dropout rate for attention softmaxes and intermediate embeddings.
        bag_of_embeddings (bool, default: False): Adds a parallel bag of embeddings layer.
            Concats to the final document embedding.
    '''

    def __init__(self,
                 embedding_matrix,
                 task_dict,
                 max_words_per_line,
                 max_lines,
                 device,
                 att_dim_per_head=50,
                 att_heads=8,
                 att_dropout=0.1,
                 bag_of_embeddings=False,
                 ):

        super(MTHiSAN, self).__init__()
        self.is_transformer = False
        self.task_dict = task_dict
        self.n_classes = {task: value["n_classes"] for task, value in self.task_dict.items()}
        self.tasks = list(task_dict.keys())


        self.max_words_per_line = max_words_per_line
        self.max_lines = max_lines
        self.max_len = max_lines * max_words_per_line
        self.att_dim_per_head = att_dim_per_head
        self.att_heads = att_heads
        self.att_dim_total = att_heads * att_dim_per_head
        

        embedding_matrix[0] = 0
        embedding_matrix = embedding_matrix.to(device)

        self.embedding = nn.Embedding.from_pretrained(
                         torch.tensor(embedding_matrix, dtype=torch.float),
                         freeze=False, padding_idx=0)
        self.word_embed_drop = nn.Dropout(p=att_dropout)
        # Q, K, V, and other layers for word-level self-attention
        self.word_q = nn.Linear(embedding_matrix.shape[1], self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_q.weight)
        self.word_q.bias.data.fill_(0.0)
        self.word_k = nn.Linear(embedding_matrix.shape[1], self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_k.weight)
        self.word_k.bias.data.fill_(0.0)
        self.word_v = nn.Linear(embedding_matrix.shape[1], self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_v.weight)
        self.word_v.bias.data.fill_(0.0)
        self.word_att_drop = nn.Dropout(p=att_dropout)

        # target vector and other layers for word-level target attention
        self.word_target_drop = nn.Dropout(p=att_dropout)
        self.word_target = nn.Linear(1, self.att_dim_total, bias=False)
        torch.nn.init.uniform_(self.word_target.weight)
        self.line_embed_drop = nn.Dropout(p=att_dropout)

        # Q, K, V, and other layers for line-level self-attention
        self.line_q = nn.Linear(self.att_dim_total, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_q.weight)
        self.line_q.bias.data.fill_(0.0)
        self.line_k = nn.Linear(self.att_dim_total, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_k.weight)
        self.line_k.bias.data.fill_(0.0)
        self.line_v = nn.Linear(self.att_dim_total, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_v.weight)
        self.line_v.bias.data.fill_(0.0)

        # target vector and other layers for line-level target attention
        self.line_att_drop = nn.Dropout(p=att_dropout)
        self.line_target_drop =  nn.Dropout(p=att_dropout)
        self.line_target = nn.Linear(1, self.att_dim_total, bias=False)
        torch.nn.init.uniform_(self.line_target.weight)
        self.doc_embed_drop = nn.Dropout(p=att_dropout)

        # optional bag of embeddings layers
        self.boe = bag_of_embeddings
        if self.boe:
            self.boe_dense = nn.Linear(embedding_matrix.shape[1], embedding_matrix.shape[1])
            torch.nn.init.xavier_uniform_(self.boe_dense.weight)
            self.boe_dense.bias.data.fill_(0.0)
            self.boe_drop = nn.Dropout(p=0.5)

        # dense classification layers
        # for Multitask
        self.fcs = nn.ModuleDict()
        for task, task_values in task_dict.items():
            in_size = self.att_dim_total
            if self.boe:
                in_size += embedding_matrix.shape[1]
            fc = nn.Linear(in_features=in_size, out_features=task_values["n_classes"])
            torch.nn.init.xavier_uniform_(fc.weight)
            fc.bias.data.fill_(0.0)
            self.fcs[task] = fc
        self.softmax = nn.LogSoftmax(dim=1) # Logsoftmax for NLLLoss later on
        



    def _split_heads(self, x):
        '''
        Splits the final dimension of a tensor into multiple heads for multihead attention.

        Args:
            x (torch.tensor): Float tensor of shape [batch_size x seq_len x dim].

        Returns:
            torch.tensor: Float tensor of shape [batch_size x att_heads x seq_len x att_dim_per_head].
            Reshaped tensor for multihead attention.
        '''

        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.att_heads, self.att_dim_per_head)
        return torch.transpose(x, 1, 2)

    def _attention(self, q, k, v, drop=None, mask_q=None, mask_k=None, mask_v=None):
        '''
        Flexible attention operation for self and target attention.

        Args:
            q (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim1].
            k (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim1].
            v (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim2].
                NOTE: q and k must have the same dimension, but v can be different.
            drop (torch.nn.Dropout): Dropout layer.
            mask_q (torch.tensor): Boolean tensor of shape [batch x seq_len].
            mask_k (torch.tensor): Boolean tensor of shape [batch x seq_len].
            mask_v (torch.tensor): Boolean tensor of shape [batch x seq_len].

        Returns:
            None
        '''

        # generate attention matrix
        # batch x heads x seq_len x seq_len
        scores = torch.matmul(q, torch.transpose(k, -1, -2)) / math.sqrt(q.size(-1))

        # this masks out empty entries in the attention matrix
        # and prevents the softmax function from assigning them any attention
        if mask_q is not None:
            mask_q = torch.unsqueeze(mask_q, 1)
            mask_q = torch.unsqueeze(mask_q, -2)
            padding_mask = torch.logical_not(mask_q)
            scores -= 1.e7 * padding_mask.float()

        # normalize attention matrix
        weights = F.softmax(scores, -1)                                          # batch x heads x seq_len x seq_len

        # this removes empty rows in the normalized attention matrix
        # and prevents them from affecting the new output sequence
        if mask_k is not None:
            mask_k = torch.unsqueeze(mask_k, 1)
            mask_k = torch.unsqueeze(mask_k, -1)
            weights = torch.mul(weights, mask_k.type(weights.dtype))

        # optional attention dropout
        if drop is not None:
            weights = drop(weights)

        # use attention on values to generate new output sequence
        result = torch.matmul(weights, v)                                        # batch x heads x seq_len x dim2

        # this applies padding to the entries in the output sequence
        # and ensures all padded entries are set to 0
        if mask_v is not None:
            mask_v = torch.unsqueeze(mask_v, 1)
            mask_v = torch.unsqueeze(mask_v, -1)
            result = torch.mul(result, mask_v.type(result.dtype))

        return result

    def forward(self, docs, return_embeds=False):
        '''
        Flexible attention operation for self and target attention.

        Args:
            q (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim1].
            k (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim1].
            v (torch.tensor): Float tensor of shape [batch x heads x seq_len x dim2].
                NOTE: q and k must have the same dimension, but v can be different.
            drop (torch.nn.Dropout): Dropout layer.
            mask_q (torch.tensor): Boolean tensor of shape [batch x seq_len].
            mask_k (torch.tensor): Boolean tensor of shape [batch x seq_len].
            mask_v (torch.tensor): Boolean tensor of shape [batch x seq_len].

        Returns:
            None
        '''
        # Addition: store outputs
        fw_outputs = {}

        # bag of embeddings operations if enabled
        if self.boe:
            mask_words = (docs != 0)
            words_per_line = mask_words.sum(-1)
            max_words = words_per_line.max()
            mask_words = torch.unsqueeze(mask_words[:, :max_words], -1)
            docs_input_reduced = docs[:, :max_words]
            word_embeds = self.embedding(docs_input_reduced)
            word_embeds = torch.mul(word_embeds, mask_words.type(word_embeds.dtype))
            bag_embeds = torch.sum(word_embeds, 1)
            bag_embeds = torch.mul(bag_embeds,
                                   1.0 / torch.unsqueeze(words_per_line, -1).type(bag_embeds.dtype))
            bag_embeds = torch.tanh(self.boe_dense(bag_embeds))
            bag_embeds = self.boe_drop(bag_embeds)

        # reshape into batch x lines x words
        docs = docs[:, :self.max_len]
        docs = docs.reshape(-1, self.max_lines, self.max_words_per_line)          # batch x max_lines x max_words

        # generate masks for word padding and empty lines
        # remove extra padding that exists across all documents in batch
        mask_words = (docs != 0)                                                # batch x max_lines x max_words
        words_per_line = mask_words.sum(-1)                                     # batch x max_lines
        max_words = words_per_line.max()                                        # hereon referred to as 'words'
        num_lines = (words_per_line != 0).sum(-1)                               # batch
        max_lines = num_lines.max()                                             # hereon referred to as 'lines'
        docs_input_reduced = docs[:, :max_lines, :max_words]                      # batch x lines x words
        mask_words = mask_words[:, :max_lines, :max_words]                        # batch x lines x words
        mask_lines = (words_per_line[:, :max_lines] != 0)                        # batch x lines

        # combine batch dim and lines dim for word level functions
        # also filter out empty lines for speedup and add them back in later
        batch_size = docs_input_reduced.size(0)
        docs_input_reduced = docs_input_reduced.reshape(
                             batch_size*max_lines, max_words)                    # batch*lines x words
        mask_words = mask_words.reshape(batch_size*max_lines, max_words)         # batch*lines x words
        mask_lines = mask_lines.reshape(batch_size*max_lines)                   # batch*lines
        docs_input_reduced = docs_input_reduced[mask_lines]                     # filtered x words
        mask_words = mask_words[mask_lines]                                     # filtered x words
        batch_size_reduced = docs_input_reduced.size(0)                         # hereon referred to as 'filtered'

        # word embeddings
        word_embeds = self.embedding(docs_input_reduced)                        # filtered x words x embed
        word_embeds = self.word_embed_drop(word_embeds)                         # filtered x words x embed

        # word self-attention
        word_q = F.elu(self._split_heads(self.word_q(word_embeds)))             # filtered x heads x words x dim
        word_k = F.elu(self._split_heads(self.word_k(word_embeds)))             # filtered x heads x words x dim
        word_v = F.elu(self._split_heads(self.word_v(word_embeds)))             # filtered x heads x words x dim
        word_att = self._attention(word_q, word_k, word_v,
                                   self.word_att_drop, mask_words,
                                   mask_words, mask_words)                        # filtered x heads x words x dim

        # word target attention
        word_target = self.word_target(word_att.new_ones((1, 1)))
        word_target = word_target.view(
                      1, self.att_heads, 1, self.att_dim_per_head)                 # 1 x heads x 1 x dim
        line_embeds = self._attention(word_target, word_att, word_att,
                                      self.word_target_drop, mask_words)                         # filtered x heads x 1 x dim
        line_embeds = line_embeds.transpose(1, 2).view(
                      batch_size_reduced, 1, self.att_dim_total).squeeze(1)       # filtered x heads*dim
        line_embeds = self.line_embed_drop(line_embeds)                         # filtered x heads*dim

        # add in empty lines that were dropped earlier for line level functions
        line_embeds_full = line_embeds.new_zeros(
                           batch_size*max_lines, self.att_dim_total)             # batch*lines x heads*dim
        line_embeds_full[mask_lines] = line_embeds
        line_embeds = line_embeds_full
        line_embeds = line_embeds.reshape(
                      batch_size, max_lines, self.att_dim_total)                  # batch x lines x heads*dim
        mask_lines = mask_lines.reshape(batch_size, max_lines)                   # batch x lines

        # line self-attention
        line_q = F.elu(self._split_heads(self.line_q(line_embeds)))             # batch x heads x lines x dim
        line_k = F.elu(self._split_heads(self.line_k(line_embeds)))             # batch x heads x lines x dim
        line_v = F.elu(self._split_heads(self.line_v(line_embeds)))             # batch x heads x lines x dim
        line_att = self._attention(line_q, line_k, line_v,
                                   self.line_att_drop, mask_lines,
                                   mask_lines, mask_lines)                       # batch x heads x lines x dim

        # line target attention
        line_target = self.line_target(line_att.new_ones((1, 1)))
        line_target = line_target.view(1, self.att_heads,
                                       1, self.att_dim_per_head)                 # 1 x heads x 1 x dim
        doc_embeds = self._attention(line_target, line_att, line_att,
                                     self.line_target_drop, mask_lines)                          # batch x heads x 1 x dim
        doc_embeds = doc_embeds.transpose(1, 2).view(
                     batch_size, 1, self.att_dim_total).squeeze(1)                # batch x heads*dim
        doc_embeds = self.doc_embed_drop(doc_embeds)                            # batch x heads*dim

        # if bag of embeddings enabled, concatenate to hisan output
        if self.boe:
            doc_embeds = torch.cat([doc_embeds, bag_embeds], 1)                   # batch x heads*dim+embed

        ## Adapted:
        # generate logits + output probs for each task
        for task, fc in self.fcs.items():
            logits = fc(doc_embeds)                                      # batch x num_classes
            output_probs = self.softmax(logits)
            fw_outputs[task] = output_probs
        if return_embeds:
            return fw_outputs, doc_embeds
        return fw_outputs