r'''
Module to retrive a batch-wrapped torchtext
iterator containing natural language data.
The file first tokenizes English language
data using spaCy and a torchtext Field.
Then the data is wrapped in a torchtext 
TabularDataset which is then passed into
a torchtext BPTTIterator and this is
finally wrapped for ease of use in retrieving
a single batch for training.
'''
# standard library imports
from __future__ import absolute_import
import os
import sys
import csv
import json
import logging

# third party imports
import spacy
import pandas as pd
from torchtext.data import Field, TabularDataset, BPTTIterator
from torchtext.datasets import WikiText2
from termcolor import cprint
from easydict import EasyDict as edict

# internal importss
from dataloaders.base_dl import BaseDL
from utils.batch import Batch


max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


class WikiText2DL(BaseDL):
    '''
    Class to retrieve the WikiText2 data for a text agent
    in the agents directory of the project.

    Arguments
    ---------
        config : obj
            Configuration object.
        device : str
            Device to run data loading.
    '''

    def __init__(self, config, device):
        super(WikiText2DL, self).__init__(config, device)
        ''' Initializes an instance of the GANTextDL class. '''
        try:
            self.process = self.config.process_data
        except Exception:
            print('No processing being used for dataloading')
            pass

        # if en_core_web_sm is not installed, run the script to install
        try:
            self.spacy_en = spacy.load('en')
        except OSError:
            cprint('INFO :: DOWNLOADING SPACY ENGLISH LANGUAGE.', 'yellow')
            os.system('python3 -m spacy download en')
            self.spacy_en = spacy.load('en')
    
    def tokenizer(self, x):
        ''' Tokenizes text data. '''
        return [tok.text for tok in self.spacy_en.tokenizer(x)]
    
    def process_data(self):
        ''' Processes the data before loading. Not implemented for this. '''
        pass

    def get_data(self):
        '''
        Retrieves data in a format that can
        be used in training by loading in batches.

        Returns
        -------
            obj
                Object loaded with language data.
            obj
                Torchtext data iterator.
            int
                Vocab size in the text dataset.
            obj
                Field object from Torchtext.
            obj
                Vocabulary taken from Torchtext Field.
        '''
        TEXT = Field(tokenize=self.tokenizer, lower=True)

        train, valid, test = WikiText2.splits(TEXT)

        TEXT.build_vocab()
        vocab_size = len(TEXT.vocab)

        train_iter, valid_iter = BPTTIterator.splits(
            (train, valid),
            batch_size=self.config.batch_size,
            bptt_len=8,
            device=self.device,
            repeat=False
        )

        train_loader = Batch(dl=train_iter, x_var='text')
        valid_loader = Batch(dl=valid_iter, x_var='text')

        print(len(train_loader))

        data_dict = edict({
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'train_iter': train_iter,
            'vocab_size': vocab_size,
            'vocab': TEXT.vocab
        })

        return data_dict