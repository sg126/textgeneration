# third party imports
from nltk.translate.bleu_score import corpus_bleu

# internal imports
from metrics.base_metric import BaseMetric


class BLEU(BaseMetric):
    '''
    Class to assign a BLEU score to a model.
    Implements the BaseMetric class.

    Arguments
    ---------
        model : obj
            Language model to score.
        candidates : array_like
            List of candidate sentences to score.
        references : array_like
            List of reference sentences to score candidates against.
    '''

    def __init__(self, model, candidates, references):
        '''
        Initializes an instance of the BLEU class. '''
        super(BLEU, self).__init__(model)
        
        self.candidates = candidates
        self.references = references
        self.score = 0.0
    
    def get_score(self):
        '''
        Computes the BLEU score for the model.

        Returns
        -------
            float
                BLEU score
        '''
        self.score = corpus_bleu(self.references, self.candidates)
        return self.score