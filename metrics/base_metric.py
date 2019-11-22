r'''
Module containing a BaseMetric class
to be extended by all classes in the metrics
directory. Used for scoring models in a simple way,
all "metric" classes must return a score of some kind.
'''
# standard library imports
from __future__ import absolute_import, print_function


class BaseMetric:
    '''
    Base class for assigning a score for a model
    using some defined criteria.

    Arguments
    ---------
        model : obj
            Model to evaluate.
    '''

    def __init__(self, model):
        '''
        Initializes an instance of the BaseMetric class.
        '''
        self.model = model
        if self.model is None:
            raise AttributeError('Error :: Language model not passed to Metric\'s constructor')
    
    def get_score(self):
        '''
        Computes the score for the intended metric.

        Returns
        -------
            float
                Score for the implementing class's intended metric.
        '''
        raise NotImplementedError