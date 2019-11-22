# standard library imports
from __future__ import absolute_import
import sys
import csv
import json
import logging
import random
from random import shuffle
from os import getcwd, system

# third party imports
import spacy
import nltk
import pandas as pd
from torchtext.data import Field, TabularDataset, BPTTIterator
from termcolor import cprint
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict

nltk.download('wordnet')
from nltk.corpus import wordnet

# internal imports
from dataloaders.base_dl import BaseDL
from utils.batch import Batch
from utils.batch import MultiColumnBatch


max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)



class NavigationInstructionsDL(BaseDL):
    '''
    Class to retrieve navigational instruction data for a text agent.

    Arguments
    ---------
        config : obj
            Configuration object
        device : str
            Device to run data loading.
    '''

    def __init__(self, config, device):
        super(NavigationInstructionsDL, self).__init__(config, device)
        ''' Initializes an instance of the NavigationInstructionsDL class. '''
        try:
            self.process = self.config.process_data
            cprint('INFO :: Data processing enabled', 'yellow')
        except Exception:
            cprint('INFO :: No data processing', 'yellow')
            pass

        self.dataroot = getcwd() + '/data/'
        self.raw_dataroot = self.dataroot + 'raw/navinstr_raw.csv'
        self.proc_dataroot = self.dataroot + 'processed/navinstr.csv'

        try:
            self.spacy_en = spacy.load('en')
        except OSError:
            cprint('INFO :: DOWNLOADING SPACY ENGLISH LANGUAGE', 'yellow')
            system('python3 -m spacy download en')
            self.spacy_en = spacy.load('en')
        
        if self.process:
            self.process_data()
    
    def process_data(self):
        ''' Processes data before loading. '''
        # read in to dataframe
        df = pd.read_csv(self.raw_dataroot, index_col=0)

        # apply data augmentation and add to a new dataframe
        new_df = df['A'].apply(nltk.word_tokenize)
        new_df = new_df.to_frame()
        new_df = new_df['A'].apply(self.synonym_replacement).apply(lambda x: ' '.join(x))
        new_df = new_df.to_frame()

        # merge the augmented dataframe and original to a new dataframe
        result_df = pd.concat([df, new_df], axis=0)

        # extract all nouns, verbs, and adjectives and add a column to the DF
        result_df['B'] = result_df['A'].apply(nltk.word_tokenize)
        correct_pos = lambda pos: pos[:2] == 'VB' or pos[:2] == 'NN' or pos[:2] == 'JJ'
        result_df['B'] = result_df['B'].apply(lambda x: [word for (word, pos) in nltk.pos_tag(x) if correct_pos(pos)])

        # rename columns and save to csv file
        result_df.columns = ['text', 'key']
        result_df.to_csv(self.proc_dataroot, index=False)
        cprint('Data processing complete', 'green')
    
    def synonym_replacement(self, words):
        ''' Run synonym replacement on the given words. '''
        new_words = words.copy()
        random_word_list = list(set([word for word in words]))
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= 1:
                break
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words
    
    def get_synonyms(self, word):
        ''' Retrieves a list of synonyms for the given word. '''
        synonyms = set()
        for s in wordnet.synsets(word):
            for l in s.lemmas():
                synonym = l.name().replace('_', ' ').replace('-', ' ').lower()
                synonym = ''.join([char for char in synonym if char in 'abcdefghijklmnopqrstuvwxyz'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    
    def tokenizer(self, x):
        ''' Tokenizes text data. '''
        return [tok.text for tok in self.spacy_en.tokenizer(x)]
    
    def get_data(self):
        '''
        Retrieves data in a format that can
        be used in training by loading in batches.

        Returns
        -------
            obj
                Dictionary of data-related information.
        '''
        TEXT = Field(tokenize=self.tokenizer, lower=True)
        KEY = Field(sequential=False, use_vocab=False)

        # read processed data to csv file
        df = pd.read_csv(self.proc_dataroot)
        
        # split dataset into train and validation and save to seperate files
        train, valid = train_test_split(df, test_size=0.2)
        train.to_csv(getcwd() + '/data/processed/navinstr_train.csv', index=False)
        valid.to_csv(getcwd() + '/data/processed/navinstr_valid.csv', index=False)

        datafields = [
            ('text', TEXT),
            ('key', KEY)
        ]

        train_set, valid_set = TabularDataset.splits(
            path=getcwd() + '/data/processed',
            train='navinstr_train.csv',
            validation='navinstr_valid.csv',
            format='csv',
            fields=datafields,
            skip_header=True
        )

        TEXT.build_vocab(train_set, valid_set)
        vocab_size = len(TEXT.vocab)

        # torchtext backprop through time iterator
        train_iter, valid_iter = BPTTIterator.splits(
            (train_set, valid_set),
            batch_size=self.config.batch_size,
            bptt_len=8,
            device=self.device,
            repeat=False,
            shuffle=True
        )

        # train_loader = Batch(dl=train_iter, x_var='text')
        # valid_loader = Batch(dl=valid_iter, x_var='text')
        train_loader = MultiColumnBatch(dl=train_iter, x_var='text', y_var='key')
        valid_loader = MultiColumnBatch(dl=valid_iter, x_var='text', y_var='key')

        data_dict = edict({
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'train_iter': train_iter,
            'vocab_size': vocab_size,
            'vocab': TEXT.vocab
        })

        return data_dict