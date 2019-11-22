# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Transformer(nn.Module):
    '''
    Class to define a transformer architecture to generate text.

    Arguments
    ---------
        config : obj
            Configuration object that contains information
            for training (eg. hyperparameters)
        vocab_size : int
            Size of the vocab in the specified data.
        _device : str           (default='cpu')
            Device to use for tensor computations (cuda or cpu)
        _cuda : bool            (default=False)
            If True, send tensors to GPU.
    '''

    def __init__(self, config, vocab_size, _device='cpu', _cuda=False):
        ''' Initializes an instance of the Transformer class. '''
        super(Transformer, self).__init__()

        self.config = config
        
        self._device = device
        self._cuda = _cuda

        self.use_recurrent = self.config.use_recurrent

        self.embedding_size = self.config.embedding_size
        self.hidden_size = self.config.hidden_size
        self.context_size = self.config.context_size
        self.vocab_size = self.vocab_size
        self.batch_size = self.config.batch_size
        self.max_seq_len = self.config.max_seq_len
        self.dropout_prob = self.config.dropout_prob
        self.num_layers = self.config.num_layers
        self.bidirectional = self.config.bidirectional
        
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.transformer1 = nn.Transformer(
            d_model=512, nhead=8,
            num_encoder_layers=6, num_decoder_layers=6,
            dim_feedforward=2048, dropout=self.dropout_prob
        )
        self.lstm = nn.LSTM(
            input_size=10, hidden_size=10,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, self.batch_size)

    def forward(self, x):
        ''' Performs a single forward pass on the data. '''
        embed = self.embeddings(x)
        embed = embed.view(-1, 1, self.embedding_size)
        h0, c0 = self.init_hidden(embed.shape[0])
        if self._cuda:
            embed = embed.to(self._device)
            h0 = h0.to(self._device)
            c0 = c0.to(self._device)
        output = self.transformer1()
        if self.use_recurrent:
            output, (h0, c0) = self.lstm()
            output = self.fc(output.view(-1, self.hidden_size))
        output = F.softmax(output, dim=1)
        return output, (h0, c0)
    
    def init_hidden(self, embed_size):
        ''' Initializes hidden (h0, c0) for LSTM layers. '''
        first_dim_size = 1
        if self.bidirectional:
            first_dim_size *= 2
        first_dim_size *= self.num_layers
        h0 = torch.zeros(first_dim_size, embed_size, self.hidden_size)
        c0 = torch.zeros(first_dim_size, embed_size, self.hidden_size)
        return h0, c0