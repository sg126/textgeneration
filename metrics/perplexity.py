# standard library imports
import math

# third party imports


# internal imports
from metrics.base_metric import BaseMetric


class Perplexity(BaseMetric):
    '''
    Class to calculate perplexity for a language model.
    Implements the BaseMetric class.

    Arguments
    ---------
        model : obj
            Language model to score.
        candidates : array_like
            List of candidate sentences to score.
    '''

    def __init__(self, model, candidates):
        ''' Initializes an instance of the Perplexity class. '''
        super(Perplexity, self).__init__(model)

        self.model = model
        self.candidates = candidates
        self.n = n
    
    def perplexity(self, text):
        '''
        '''
        return pow(2.0, self.entropy(text))

    def get_score(self):
        '''
        Computes the perplexity for the given model.

        Returns
        -------
            float
                Perplexity score
        '''
        self.probability_log_sum = self.calculate_bigram_perplexity()
        return self.probability_log_sum