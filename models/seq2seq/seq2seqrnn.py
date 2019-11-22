# third party imports
import torch
import torch.nn as nn


class Encoder(nn.Module):
    '''

    Arguments
    ---------
        config : obj
            Configuration object.
    '''

    def __init__(self, config):
        ''' Initializes an instance of the Encoder class. '''
        super(Encoder, self).__init__()

        self.config = config
        
        self.emb_size = self.config.emb_size
        self.hidden_size = self.config.hidden_size
        self.dropout_prob = self.config.dropout_prob
    
    def forward(self, x):
        ''' Performs a single forward pass on the data. '''
        pass

class Decoder(nn.Module):
    '''

    Arguments
    ---------
        config : obj
            Configuration object.
    '''

    def __init__(self, config):
        ''' Initializes an instance of the Decoder class. '''
        super(Decoder, self).__init__()

        self.config = config

        self.emb_size = self.config.emb_size
        self.hidden_size = self.config.hidden_size
        self.dropout_prob = self.config.dropout_prob
    
    def forward(self, x):
        ''' Performs a single forward pass on the data. '''
        pass