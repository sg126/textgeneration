# third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    '''
    Class to contain architecture for a Generator in a GAN. Constructs
    fake data that a discriminator should be fooled to think is real.

    Arguments
    ---------
        config : obj
            Configuration object that contains
            information such as hyperparameters.
        vocab_size : int
            Size of the vocab in the specified data.
        _device : str               (default='cpu')
            Device to use for tensor computations (cuda or cpu)
        _cuda : bool                (default=False)
            If True, send tensors to GPU.
    '''

    def __init__(self, config, vocab_size, _device='cpu', _cuda=False):
        ''' Initializes an instance of the Generator class. '''
        super(Generator, self).__init__()

        self.config = config

        self._device = _device
        self._cuda = _cuda

        self.embedding_size = self.config.embedding_size
        self.hidden_size = self.config.hidden_size
        self.context_size = self.config.context_size
        self.vocab_size = vocab_size
        self.batch_size = self.config.batch_size
        self.max_seq_len = self.config.max_seq_len
        self.dropout_prob = self.config.dropout_prob
        self.num_layers = self.config.num_layers
        self.bidirectional = self.config.bidirectional

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            self.embedding_size, self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout_prob,
            bidirectional=self.config.bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, self.batch_size)
    
    def forward(self, x):
        '''
        Method to send the data forward through the network.

        Arguments
        ---------
            x : array_like
                Input data to be transformed by the network.

        Returns
        -------
            array_like
                Transformed list of values after being passed through the network.
            array_like
                Hidden tensor also returned.
        '''
        embed = self.embeddings(x)
        embed = embed.view(-1, 1, self.embedding_size)
        h0, c0 = self.init_hidden(embed.shape[0])
        if self._cuda:
            embed = embed.to(self._device)
            h0 = h0.to(self._device)
            c0 = c0.to(self._device)
        temp, hidden = self.lstm(embed, (h0, c0))
        output = self.dropout(temp)
        output = output.contiguous()
        output = self.fc(output.view(-1, self.hidden_size))
        output = F.softmax(output, dim=1)
        return output, hidden
    
    def init_hidden(self, embed_size):
        '''
        Retrieves a hidden torch Variable using the given size.

        Arguments
        ---------
            embed_size : int    
                Size of a hidden tensor.
        
        Returns
        -------
            array_like
                Tensor for hidden layers in the network (used in LSTM).
            array_like
                Another tensor for the LSTM.
        '''
        first_dim_size = 1
        if self.bidirectional:
            first_dim_size *= 2
        first_dim_size *= self.num_layers
        h0 = torch.zeros(first_dim_size, embed_size, self.hidden_size)
        c0 = torch.zeros(first_dim_size, embed_size, self.hidden_size)
        return h0, c0

    def predict(self, words, vocab, num_samples=1, topk=10):
        '''
        Outputs natural language sentence(s) given
        a set of words.

        Arguments
        ---------
            words : array_like
                List of words to be used in the sentence.
            vocab : obj
                Vocabulary object containing all the words in the dataset.
            num_samples : int           (default=1)
                Number of samples to output.
        
        Returns
        -------
            array_like
                List of generated text samples.
        '''
        stopwords = ['<unk>', '<pad>', '.']
        self.eval()

        h0, c0 = self.init_hidden(embed_size=64)
        if self.cuda:
            h0 = h0.to(self._device)
            c0 = c0.to(self._device)
        for w in words:
            idx = torch.tensor([vocab.stoi[w]]).to(self._device)
            output, hidden = self.forward(idx)

        _, top_idx = torch.topk(output[0], k=topk)
        choices = top_idx.tolist()
        for c in choices:
            if c != 0:
                random_choice = np.random.choice(choices[0])
        words.append(vocab.itos[random_choice])

        for _ in range(10):
            idx = torch.tensor([[random_choice]]).to(self._device)
            output, (h0, c0) = self.forward(idx)

            _, top_idx = torch.topk(output[0], k=topk)
            choices = top_idx.tolist()
            choice = np.random.choice(choices[0])
            words.append(vocab.itos[choice])
        
        result = [word for word in words if word not in stopwords]
        return ' '.join(result)