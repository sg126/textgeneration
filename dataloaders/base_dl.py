r'''
Module to define a base dataloader class
to be used by implementing classes in the
dataloaders directory of this project.
'''
# standard library imports
from __future__ import absolute_import
import logging


class BaseDL:
    '''
    Base dataloader class which outlines
    the basic functions needed to load
    natural language data for use in training.

    Arguments
    ---------
        config : obj
            Configuration object.
        device : str
            Device to use for dataloading operations.
    '''

    def __init__(self, config, device):
        ''' Initializes an instance of the BaseDL class. '''
        self.config = config
        self.logger = logging.getLogger('DataLoader')
        self.device = device

    def process_data(self):
        ''' Processes the data before loading. '''
        raise NotImplementedError
    
    def get_data(self):
        '''
        Retrieves data in a format that can
        be used in training by loading in batches.

        Returns
        -------
            obj
                Object loaded with language data.
        '''
        raise NotImplementedError