# third party import
import torch


class Batch:
    '''
    Class to wrap a torchtext iterator
    and retrieve a single batch of data during iteration.

    Arguments
    ---------
        dl : obj
            Data loading object.
        x_var : str
            Name of the column in the given dataloader to retrieve
    '''

    def __init__(self, dl, x_var):
        ''' Initializes an instance of the Batch class. '''
        self.dl = dl
        self.x_var = x_var
    
    def __iter__(self):
        '''
        Retrieves the next item in this data loading object.

        Returns
        -------
            obj
                One batch of data.
        '''
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            yield x
    
    def __len__(self):
        '''
        Computes the length of the data.

        Returns
        -------
            int
                Length of the data.
        '''
        return len(self.dl)


class MultiColumnBatch:
    '''
    Class to wrap a torchtext iterator
    and retrieve a single batch with multiple columns.

    Arguments
    ---------
        dl : obj
            Data loading object (torchtext iterator).
        x_var : str
            Name of column in the given dataloader to use as input.
        y_var : str
            Name of the column in the given dataloader to use for output.
    '''

    def __init__(self, dl, x_var, y_var):
        ''' Initializes an instance of the MultiColumnBatch class. '''
        self.dl = dl
        self.x_var = x_var
        self.y_var = y_var
    
    def __iter__(self):
        '''
        Retrives the next item.

        Returns
        -------
            array_like : (obj, obj)
                One batch of data.
        '''
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            if self.y_var is None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_var], dim=1).float()
            else:
                y = torch.zeros((1))
            yield (x, y)

    def __len__(self):
        '''
        Computes the length of the data.

        Returns
        -------
            int
                Length of the data.
        '''
        return len(self.dl)