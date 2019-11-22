r'''
'''
# standard library imports
from __future__ import absolute_import
import math
import tqdm
import random
import shutil
from os import getcwd
from datetime import datetime

# third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint

# internal imports
from agents.base_agent import BaseAgent


cudnn.benchmark = True


class Seq2SeqAgent(BaseAgent):
    '''
    Class that implements BaseAgent to train
    a sequence to sequence agent.

    Arguments
    ---------
        config : obj
            Configuration object.
    '''

    def __init__(self, config):
        ''' Initializes an instance of the Seq2SeqAgent class. '''
        super().__init__(config)

        self.start = datetime.now()
        self.end = None
        self.save_filepath = getcwd() + self.config.checkpoint_dir

        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail and not self.config.cuda:
            self.logger.info('WARNING :: CUDA is available but is not being used.')
        self.cuda = self.cuda_avail and self.config.cuda

        self.manual_seed = self.config.seed
        if self.manual_seed <= 0:
            self.manual_seed = random.randint(1, 10000):
            self.logger.info('INFO :: Random seed used.')
        self.logger.info('SEED :: ', self.manual_seed)
        random.seed(self.manual_seed)

        if self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.gpu_device)
            torch.manual_seed(self.manual_seed)
            self.logger.info('CUDA in use.')
            show_gpu()
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.manual_seed)
            self.logger.info('CPU in use (not on CUDA)')
        
        # data loading

        # set epoch and batch number tracking
        self.current_epoch = 0
        self.num_epochs = self.config.num_epochs
        self.current_batch = 0

        # define models and optimizers
        self.enc_class = globals()[self.config.encoder]
        self.dec_class = globals()[self.config.decoder]
        
        self.encoder = self.

        # loss

        # send nn's to selected device

        # load checkpoint if chosen
        try:
            self.load_checkpoint(self.config.checkpoint_file)
        except AttributeError:
            pass

        # checkpoint file for saving models
        self.checkpoint_file = self.config.checkpoint_file
        self.is_best = False

        # tensorboard
        self.writer = SummaryWriter(
            log_dir=getcwd() + self.config.log_dir,
            comment='Seq2Seq'
        )

    def load_checkpoint(self, file_name='seq2seq.pth'):
        ''' Loads the checkpoint from the given file. '''
        filepath = self.config.checkpoint_dir + file_name

        try:
            self.logger.info(f'Loading checkpoint from {filepath}')
            checkpoint = torch.load(filepath)

            self.current_epoch = checkpoint['epoch']
            self.current_batch = checkpoint['batch']
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info(f'Checkpoint loaded from {filepath}\t\t::\t\t\
                Currently at batch {self.current_batch} of epoch {self.current_epoch}.')
        except OSError:
            self.logger.info(f'No checkpint at specified file path {filepath}.\
                Skipping for now.')
            self.logger.info('Checkpoint load failure most likely because filepath is incorrect \
                or this is the first training run.')
    
    def save_checkpoint(self, file_name='seq2seq.pth', is_best=False):
        '''
        Saves the model to the given file.
        
        Arguments
        ---------
            file_name : str             (default='seq2seq.pth')
                Name of the file to save the model to.
            is_best : bool              (default=False)
                If true, this is the best model to date.
        '''
        state = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'manual_seed': self.manual_seed
        }

        self.save_filepath = getcwd() + self.config.checkpoint_dir + file_name
        while True:
            try:
                torch.save(state, self.save_filepath)
                break
            except FileNotFoundError:
                cprint('ERROR :: File not found.', 'red')
                cprint('INFO :: Creating file and trying to checkpoint again', 'yellow')
                f = optn(self.save_filepath, 'w+')
        if is_best:
            shutil.copyfile(self.save_filepath, self.config.checkpoint_dir + 'best_model.pth')
    
    def run(self):
        ''' Runs the agent. '''
        try:
            losses = self.train()
            return losses
        except Exception as e:
            raise e
    
    def train(self):
        ''' Function to train the model. '''
        losses = 0
        self.start = datetime.now()
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint(file_name=self.checkpoint_file, is_best=False)
        self.end = datetime.now()
        return losses
    
    def train_one_epoch(self):
        ''' Runs a single training epoch. '''
        pass
    
    def validate(self):
        ''' Runs a cycle of validation on the model. '''
        pass
    
    def finalize(self):
        ''' Finalizes training. '''
        pass