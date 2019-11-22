# third party imports
import torch
import torch.nn as nn
from torch.autograd import Variable


class Discriminator(nn.Module):
    '''
    Class to contain architecture for a Discriminator in a GAN. Discriminates
    between real and fake data (created by another neural network).

    Arguments
    ---------
        config : 
            Configuration object that contains
            information such as hyperparameters.
        vocab_size : int
            Size of the vocab in the specified data.
        _device : str               (default='cpu')
            Device to use for tensor computations (cuda or cpu).
        _cuda : bool                (default=False)
            If True, send tensors to GPU.
    '''

    def __init__(self, config, vocab_size, _device='cpu', _cuda=False):
        ''' Initializes an instance of the Discriminator class. '''
        super(Discriminator, self).__init__()
        self.config = config

        self._device = _device
        self._cuda = _cuda

        self.embedding_size = self.config.embedding_size
        self.context_size = self.config.context_size
        self.hidden_size = self.config.hidden_size
        self.vocab_size = vocab_size
        self.batch_size = self.config.batch_size
        self.dropout_prob = self.config.dropout_prob
        self.num_layers = self.config.num_layers
        self.bidirectional = self.config.bidirectional

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(
            self.embedding_size, self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout_prob,
            bidirectional=self.bidirectional, batch_first=True
        )
        self.fc = nn.Linear(2 * 2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.out = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x):
        '''
        Method to send the data forward through the network.

        Arguments
        ---------
            x : array_like
                the input data to be transformed by the network.
            hidden : array_like
                Hidden tensor (initialized to 0).
        
        Returns
        -------
            array_like
                Transformed list of values after being passed through the network.
        '''
        embed = self.embeddings(x)
        h0 = self.init_hidden(self.batch_size)
        embed = embed.permute(1, 0, 2)
        if self._cuda:
            embed = embed.to(self._device)
            h0 = h0.to(self._device)
        _, hidden = self.gru(embed, h0)
        hidden = hidden.permute(1, 0, 2).contiguous()
        output = self.fc(hidden.view(-1, 4 * self.hidden_size))
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.out(output)
        output = torch.sigmoid(output)
        return output
    
    def init_hidden(self, batch_size=1):
        '''
        Retrieves a hidden torch Variable using the given batch size.

        Arguments
        ---------
            batch_size : int        (default=1)
                Size of a batch for initializing the hidden tensor.
        
        Returns
        -------
            array_like
                Tensor for hidden layers in the network.
        '''
        if self.bidirectional:
            h0 = Variable(torch.zeros(self.num_layers * 2, batch_size,
                    self.hidden_size).type(torch.FloatTensor))
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch_size,
                    self.hidden_size).type(torch.FloatTensor))
        return h0