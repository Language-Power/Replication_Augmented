"""
Builds dataset class
"""
import os
import json
import pandas as pd
import numpy as np
from functools import reduce
from operator import add
from copy import deepcopy
import logging
import re
import nltk

from nltk import sent_tokenize
from transformers import CamembertTokenizer
from transformers import FlaubertTokenizer
from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import LongformerTokenizer
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


import torch
from TransferSociologist.config import BaseConfig
from TransferSociologist.utils import InputError

nltk.download('punkt') 


class Dataset():
    """
    Requires a config or some necessary args to a .csv or a jsonl output of doccano
    """
    def __init__(self, config=None):
        """
        Requires either a config, or manually-passed arguments following the data config template
        """
        if config is not None:
            self.data = config.data
            self.model = config.model
        self._has_been_encoded_ = False

    def _check_necessary_args_(self):
        for nec_arg in set(self.necessary_args):
            # Check if it is in data or in model
            if (nec_arg in self.data.keys() or nec_arg in self.model.keys())==False:
                raise InputError(self.data, f"Attribute 'data' not containing necessary arguments : {self.necessary_args}")
            # TODO : Better handling of errors, raise arg which is not found properly

    def _select_tokenizer_(self):
        mods = ['CamemBert', 'FlauBert', 'Bert', 'DistilBert', 'Longformer']
        if self.model['bert_model'] == 'CamemBert':
            self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        elif self.model['bert_model'] == 'FlauBert':
            self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base')
        elif self.model['bert_model'] == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif self.model['bert_model'] == 'DistilBert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        elif  self.model['bert_model'] == 'Longformer':
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        else:
            raise NotImplementedError(f'Models supported for the moment are : {mods}')

    def _clean_labels_(self, task_type):
        if task_type == 'sequence_labelling':
            # Ensure labels are read as lists and spans are sorted
            self.df['labels'] = self.df.labels.fillna(value='[]')
            if type(self.df.iloc[0].labels) == str:
                self.df.labels = self.df.labels.apply(ev)
            self.df.labels = self.df.labels.apply(lambda x: sorted(x, key=lambda y: y[0]))
            assert type(self.df.iloc[0].labels) == list, "Invalid label format, labels\
                should be a list and look like this [[0, 260, 'label2'], [263, 595, 'label1']]"
            self.df['labels'] = self.df.labels.apply(lambda y : [x for x in y if x[2]!='RAS'])
        elif task_type == 'sentence_classification':
            # Ensure labels are read as lists and spans are sorted
            if type(self.df.iloc[0].labels) == str:
                self.df.labels = self.df.labels.apply(ev)
            self.df.labels = self.df.labels.apply(lambda x: sorted(x, key=lambda y: y[0]))
            assert type(self.df.iloc[0].labels) == list, "Invalid label format, labels\
                should be a list and look like this [[0, 260, 'label2'], [263, 595, 'label1']]"
        elif task_type  == 'text_classification':
            # Check that labels are either str or int
            assert type(self.df.iloc[0].labels) in [str, int], "Invalid label format"
            # Check for NaN:
            if self.df['labels'].isna().sum() > 0:
                logging.warning('There are NaN in the database. Automatically filling with label nul')
                self.df['labels'] = self.df.labels.fillna(value='nul')
        else:
            raise InputError(task_type,
                'task_type parameter should be sequence_labelling, sentence_classification, text_classification')


    def _compute_available_labels_(self, task_type):
        if task_type in ['sentence_classification', 'sequence_labelling']:
            self.available_labels = list(set().union(*self.df.labels.apply(lambda x: set([z[2] for z in x]))))
            if 'RAS' in self.available_labels:
                self.available_labels.remove('RAS')
            self.conversion_dict = {self.available_labels[i]: i+1 for i in range(len(self.available_labels))}
            if task_type == 'sequence_labelling':
                self.conversion_dict['O'] = 0
            elif task_type == 'sentence_classification':
                self.conversion_dict['nul'] = len(self.available_labels)
                self.conversion_dict = {i : j for j, i in enumerate(self.conversion_dict.keys())}
        elif task_type == 'text_classification':
            self.available_labels = list(self.df.labels.unique())
            self.conversion_dict = {self.available_labels[i]: i+1 for i in range(len(self.available_labels))}
        else:
            raise InputError(task_type,
                'task_type parameter should be sequence_labelling, sentence_classification, text_classification')


    def read(self, **kwargs):
        if not hasattr(self, 'data') or self.data=={}:
            self.data = kwargs
        # Ensure minimum args are given at this point:
        self.necessary_args = ['data_type', 'data_path']
        self._check_necessary_args_()
        has_gold_standard = 'gold_standard_path' in self.data.keys() and self.data['gold_standard_path'] is not None
        if self.data['data_type'] == "doccanno":
            with open(self.data['data_path'], 'r', encoding='utf-8') as f:
                file = list(map(lambda x: x.rstrip('\n'), f.readlines()))
                files = list(map(lambda x: json.loads(x), file))
                test = {str(i) : files[i] for i in range(len(files))}
                self.df = pd.read_json(path_or_buf=json.dumps(test), orient='record').transpose().drop(['id'], axis=1)
                self.df['is_gold_standard'] = [False]*self.df.shape[0]
            if has_gold_standard==True:
                with open(self.data['gold_standard_path'], 'r', encoding='utf-8') as f:
                    file = list(map(lambda x: x.rstrip('\n'), f.readlines()))
                    files = list(map(lambda x: json.loads(x), file))
                    test = {str(i) : files[i] for i in range(len(files))}
                    gs = pd.read_json(path_or_buf=json.dumps(test), orient='record').transpose().drop(['id'], axis=1)
                    gs['is_gold_standard'] = [True]*gs.shape[0]
                    self.df = pd.concat([self.df, gs])
        elif self.data['data_type'] == "csv":
            self.df = pd.read_csv(self.data['data_path'])
            self.df['is_gold_standard'] = [False]*self.df.shape[0]
            if has_gold_standard==True:
                gs = pd.read_csv(self.data['gold_standard_path'])
                gs['is_gold_standard'] = [True]*gs.shape[0]
                self.df = pd.concat([self.df, gs])
        else:
            raise InputError(self.data['data_type'], "Data type not implemented yet, try to use doccanno or csv format")
        # Do a sanity check : checking if we have columns named "labels" and "text"
        try:
            ['text', 'is_gold_standard'] in list(self.df.columns)
        except:
            raise  InputError(self.data['data_type'], "File does not have 'text', or 'is_gold_standard' columns")


    def task_encode(self, **kwargs):
        if not hasattr(self, 'model') or self.model=={}:
            # if 'test_size' in kwargs.keys():
            #     self.data['test_size'] = kwargs['test_size']
            #     kwargs.pop('test_size')
            self.model = kwargs
        # Adding necessary args and checking before going further
        self.necessary_args += ["task_type", "bert_model"]
        self._check_necessary_args_()
        # Selecting tokenizer, cleaning labels (i.e. checking type and eval if necessary)
        # Compute dictionnary of labels {label->int}, list of available labels
        self._select_tokenizer_()
        if 'pred_mode' in self.model.keys() and self.model['pred_mode']==True:
            self._pred_mode_ = True
        else:
            self._pred_mode_ = False
        if 'pred_gs' in self.model.keys() and self.model['pred_gs'] == True:
            self.pred_gs = True
        else:
            self.pred_gs = False
        if self._pred_mode_ == False or self.pred_gs==True:
            self._clean_labels_(self.model['task_type'])
            self._compute_available_labels_(self.model['task_type'])
        self.df = self.df.dropna(subset=['text'])

        if self.model['task_type'] == "text_classification":
            self.df = self.df.rename({'text': 'sents'}, axis=1)

        elif self.model['task_type'] == "sentence_classification":
            if self._pred_mode_==False or self.pred_gs==True:
                cleaned = self.df.apply(lambda x: _cleannsplit_(x, self.available_labels), axis=1)
                res = reduce(add, cleaned.values)
                sents, labels, text, span =  zip(*res)
                df2 = pd.DataFrame({'sents' : list(sents), 'labels' : list(labels), 'text':list(text), 'span':list(span)})
                self.df = pd.merge(self.df.drop(['labels'], axis=1), df2, left_on='text', right_on='text', how='right')
            else:
                cleaned = self.df.text.apply(lambda x: list(zip(sent_tokenize(x),
                                                                [x]*len(sent_tokenize(x)),
                                                                find_spans_sents_tokenize(sent_tokenize(x), x))))
                res = reduce(add, cleaned.values)
                sents, text, spans =  zip(*res)
                df2 = pd.DataFrame({'sents' : list(sents), 'text':list(text), 'spans':list(spans)})
                self.df = pd.merge(self.df, df2, left_on='text', right_on='text', how='right')

        elif self.model['task_type'] == "sequence_labelling":
            self.df['tokenized'] = self.df.text.apply(lambda x: self.tokenizer.tokenize(x))

            if self._pred_mode_==False:
                self.df['extracted_spans'] = self.df.apply(lambda x: extract_spans(x.labels, x.text), axis=1)
                self.df['spans_tokenized'] = self.df.extracted_spans.apply(lambda y: list(map(lambda x: (self.tokenizer.tokenize(x[0]), x[1]), y)))
                self.df['labels'] = self.df.apply(lambda x: align(x.tokenized, x.spans_tokenized, self.conversion_dict), axis=1)
                # A bit of cleaning : drop columns that are now useless
                self.df = self.df.drop(['extracted_spans', 'spans_tokenized'], axis=1)
            self.df = self.df.rename({'text': 'sents'}, axis=1)

        # TODO : give the choice to regularize length or use longformer!!!
        # NOW encode

    def encode_torch(self, **kwargs):
        assert self._has_been_encoded_==False, "Dataset already encoded"
        if 'test_size' in kwargs.keys():
            self.data['test_size'] = kwargs['test_size']
            kwargs.pop('test_size')
        if 'random_seed' in kwargs.keys():
            self.data['random_seed'] = kwargs['random_seed']
            kwargs.pop('random_seed')
        if not hasattr(self, 'model') or self.model=={}:
            self.model = kwargs
        # Adding necessary args and checking before going further
        self.necessary_args += ["task_type", "bert_model"]
        self._check_necessary_args_()
        # Selecting tokenizer, cleaning labels (i.e. checking type and eval if necessary)
        # Compute dictionnary of labels {label->int}, list of available label
        # Get indices of eventual gold standard that was concatenated with df
        self.df = self.df.sort_values(by='is_gold_standard', ascending=True)
        size_gs = self.df[self.df.is_gold_standard==True].shape[0]
        sentences = list(self.df['sents'].values)

        batch = self.tokenizer(
                            sentences,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation=True,          # Truncate all sentences.
                            padding=True
                    )
        input_ids = batch['input_ids']
        attention_masks = batch['attention_mask']
        # Print sentence 0, now as a list of IDs.
        logging.info(f'Original: {sentences[0]}')
        logging.info(f'Token IDs: {input_ids[0]}')
        #logging.info(f'Max sentence length: { max([len(sen) for sen in input_ids])}')
        #MAX_LEN = min(max([len(sen) for sen in input_ids]), 512)
        #logging.info(f'\nPadding/truncating all sentences to {MAX_LEN} values...')
        #logging.info(f'\nPadding token: "{self.tokenizer.pad_token}", ID: {self.tokenizer.pad_token_id}')
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        #input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
        #                        value=0, truncating="post", padding="post")

        if self._pred_mode_:
            self.pred = (input_ids, attention_masks)
            return None

        if self.model['task_type'] in ['sentence_classification', 'text_classification']:
            labels = np.array(list(map(lambda x: self.conversion_dict[x], self.df.labels.values)))
        elif self.model['task_type'] == "sequence_labelling":
            labels = deepcopy(list(self.df.labels.values))
            # don't forget to pad
            #labels = pad_sequences(labels, maxlen=MAX_LEN, dtype="long", 
            #        value=0, truncating="post", padding="post")
            # print(len(labels))
            # print(type(labels))
            dimi = len(batch['attention_mask'][0])
            # print(dimi)
            for i in range(len(labels)):
                labels[i] = labels[i]+[0]*(dimi-len(labels[i]))
                # print(labels[i])
            labels = np.array(labels)
            # ok until here, problem under
        else:
            raise NotImplementedError


        # Now train-test-split
        # test if there is a gold standard:
        has_gold_standard = 'gold_standard_path' in self.data.keys() and self.data['gold_standard_path'] is not None
        has_train_test_split = 'test_size' in self.data.keys() and self.data['test_size'] is not None
        if has_train_test_split==True:
            try:
                random_seed = self.data['random_seed']
            except:
                random_seed = 42
            train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                        random_state=random_seed, test_size=self.data['test_size'])
            # Do the same for the masks.
            train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                        random_state=random_seed, test_size=self.data['test_size'])
        elif has_gold_standard==True:  # Prioritary over ttsplit
            print("Using gold standard")
            train_inputs, validation_inputs = input_ids[:-size_gs], input_ids[-size_gs:]
            train_labels, validation_labels = labels[:-size_gs], labels[-size_gs:]
            train_masks, validation_masks = attention_masks[:-size_gs], attention_masks[-size_gs:]
        else:
            raise InputError(self.data, 'Given config does not contain either test_size or gold_standard_path parameter')

        self._has_been_encoded_ = True
        logging.info('Dataset has been encoded. Access train and test sets with\
            dataset.train, dataset.test . dataset.train contains (train_inputs,\
            train_labels, train_masks). Same structure for test.')
        self.train = (train_inputs, train_labels, train_masks)
        self.test = (validation_inputs, validation_labels, validation_masks)
        # return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks


def _cleannsplit_(x, available_labels):
    """
    Args:
        x : Dataset().df initialized dataset with labels and text attributes
    """
    label = x.labels
    text = x.text
    ls_texts = []
    #print(label)
    # assert len(label)>0, "No annotation"
    if len(label)==1 and label[0][2]=='RAS':
        sents = sent_tokenize(text)
        spans = find_spans_sents_tokenize(sents, text)
        ls_texts += list(map(list, zip(sents, ['nul']*len(sents), [x.text]*len(sents), spans)))
        # TODO : IMPLEMENT SPANS
    elif len(label)==0:
        sents = sent_tokenize(text)
        spans = find_spans_sents_tokenize(sents, text)
        ls_texts += list(map(list, zip(sents, ['nul']*len(sents), [x.text]*len(sents), spans)))
        # TODO : IMPLEMENT SPANS

    else:
        label2 = []
        if label[0][0]>0:
            label2.append([0, label[0][0], 'nul'])
        for i in range(len(label)-1):
            if label[i][1] <= label[i+1][0]: # Case of chevauchement
                label[i+1][0] = label[i+1][0]-1
            if label[i+1][0] - label[i][1] > 2:
                nup = [label[i][1], label[i+1][0], 'nul']
                label2.append(label[i])
                label2.append(nup)
            else:
                label2.append(label[i])
        label2.append(label[-1])
        if label[-1][1] < len(text):
            label2.append([label[-1][1], len(text), 'nul'])
        label = label2
#    return label
        for nuplet in label:
            if nuplet[2] in available_labels:
                ls_texts.append([text[nuplet[0]:nuplet[1]], nuplet[2], x.text, [nuplet[0], nuplet[1]]])
            else:
                t = text[nuplet[0]:nuplet[1]]
                sents = sent_tokenize(t)
                for s in sents:
                    ls_texts.append([s, 'nul', x.text, [nuplet[0], nuplet[1]]])
    return ls_texts


def find_spans_sents_tokenize(sents, text):
    spans = []
    last_span = 0
    for sent in sents:
        l = len(sent)
        start_span = text.find(sent, last_span)
        stop_span = start_span+l
        spans.append([start_span, stop_span])
        last_span = stop_span
    return spans


# Testing if spans from labels are ok : 
def extract_spans(span_ls, text):
    """
    Args :
        spans_ls : list
            list of spans
        text : string
            the text string to retrieve the spans in
    Returns :
        res : list of strings
            the strings corresponding to the tags
    """
    res = []
    for i in span_ls:
        try:
            if bool(re.match(r"[a-z]{1}|[A-Z]{1}", text[i[0]-1])):
                res.append((text[i[0]-1:i[1]-1], i[2]))
                #print('matched :\t new {0} \t old: {1}'.format(text[i[0]-1:i[1]-1], text[i[0]:i[1]]))
            elif bool(re.match(r"[.;,:?]{1}", text[i[1]-1])):
                res.append((text[i[0]:i[1]-1], i[2]))
                #print('matched :\t new {0} \t old: {1}'.format(text[i[0]:i[1]-1], text[i[0]:i[1]]))
            #elif bool(re.match(r"[ ]{1}[.;,:?]{1}", text[i[1]-1])):
            #    res.append((text[i[0]:i[1]-2], i[2]))
            #    print('matched :\t new {0} \t old: {1}'.format(text[i[0]:i[1]-1], text[i[0]:i[1]]))
            else:
                res.append((text[i[0]:i[1]], i[2]))
        except:
            #print('len : {2}, i : {3}, old : {0}, new : {1}'.format(text[i[0]:i[1]], 
            #                                                        text[i[0]:i[1]-4], 
            #                                                        len(text[i[0]:i[1]]), 
            #                                                        i[1]-i[0]))
            res.append((text[i[0]:i[1]-4], i[2]))
    return res

def align(tokenized_text, tokenized_span, conversion_dict):
    # Supposing that tokenized_span exist : 
    if len(tokenized_span)==0:
        return [conversion_dict['O']]*len(tokenized_text)
    else:
        base_mask = [conversion_dict['O']]*len(tokenized_text)
        total_tag = 0
        catched_tags = 0
        for span in tokenized_span:
            found_tag = False
            sequence, tag = span
            n = len(sequence)
            for i in range(len(tokenized_text)-n):
                if tokenized_text[i:i+n] == sequence:
                    base_mask[i:i+n] = [conversion_dict[tag]]*n
                    found_tag = True
                    catched_tags = catched_tags+1
            if found_tag == False:
                #print('Tag : {} not found'.format(' '.join(sequence)) )
                #print('Trying to catch border bugs...')
                for j in range(len(tokenized_text)-n-1):
                    if tokenized_text[j+1:j+n-1] == sequence[1:-1]:
                        base_mask[j:j+n] = [conversion_dict[tag]]*n
                        found_tag = True
                        #print('Catched tag! : {}'.format(' '.join(tokenized_text[j:j+n])))
                        print("Tag:  correctly found")
                        catched_tags = catched_tags+1
                        
                if found_tag == False:
                    print('Tag : {} not found'.format(' '.join(sequence)) )
            else:
                print("Tag:  correctly found")
                catched_tags = catched_tags+1
            total_tag = total_tag+1
    # print(f'{catched_tags} / {total_tag} found overall')
    return base_mask



def ev(x):
    try:
        return eval(x)
    except:
        print('Error : {}'.format(x))