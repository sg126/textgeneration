r'''
This module contains the BaseAgent which is
a class that can be extended by all other agents
in the `/agents` directory to run training for a single
model (or set of models).
'''
# standard library imports
import logging


class BaseAgent:
    '''
    Base class to be extended by other agents
    to train various possible types of neural network
    architectures.

    Arguments
    ---------
        config : obj
            Configuration object.
    '''

    def __init__(self, config):
        ''' Initializes an instance of the BaseAgent class. '''
        self.config = config
        self.logger = logging.getLogger('Agent')
    
    def load_checkpoint(self, file_name):
        '''
        Loads the checkpoint at the specified file.

        Arguments
        ---------
            file_name : str
                Name of the file to load the checkpoint from.
        '''
        raise NotImplementedError
    
    def save_checkpoint(self, file_name='checkpoint.pth', is_best=False):
        '''
        Saves the state dicts of the model and optimizers for
        the agent's model(s) at the specified location.

        Arguments
        ---------
            file_name : str
                Name of the file to save the checkpoint to.
            is_best : bool
                If true, the checkpoint's metrics are the best so far.
        '''
        raise NotImplementedError
    
    def run(self):
        ''' Main call for training, validation, etc. '''
        raise NotImplementedError
    
    def train(self):
        ''' Training loop. '''
        raise NotImplementedError

    def train_one_epoch(self):
        ''' Runs one training epoch for the agent's model(s). '''
        raise NotImplementedError
    
    def validate(self):
        ''' Runs a cycle of validation on the model(s) for the agent. '''
        raise NotImplementedError
    
    def finalize(self):
        ''' Finalizes operations for the agent. '''
        raise NotImplementedError
    
    def evaluate(self):
        ''' Runs evaluation metrics based on the domain of the model. '''
        raise NotImplementedError
