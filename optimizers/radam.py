r'''
Module defining the rectified Adam (RAdam) optimizer
in PyTorch by extending the Optimizer class.
'''
# standard library imports
import math

# third party imports
import torch
from torch.optim.optimizer import Optimizer, required


class RAdam(Optimizer):
    '''
    Implements the rectified Adam (RAdam) optimizer introduced
    in the paper "On the Variance of the Adaptive Learning Rate and Beyond"
    by  Liu, et al. (https://arxiv.org/abs/1908.03265v1) 

    Arguments
    ---------
        params : iterable
            Iterable of parameters to optimizer
            or dicts defining parameter group
        lr : float                              (default=1e-3)
            Learning rate
        betas : tuple : (float, float)          (default=(0.9, 0.999))
            Coefficients used for computing running averages
            of gradient and its square.
        eps : float                             (default=1e-8)
            Term added to the denominator to improve
            numerical stability.
        weight_decay : int
            Weight decay (L2 penalty).
    '''

    def init(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False):
        ''' Initializes an instance of the RAdam optimizer. '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid beta value at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Invalid bet value at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(RAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def step(self, closure=None):
        '''
        Performs a single optimization step.

        Arguments
        ---------
            closure : callable              (default=None)
                A closure that reevaluates the model
                and returns the loss
        '''
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_group:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.beta.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients.')
                amsgrad = group['amsgrad']

                p_data_float = p.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_float)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_float)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_float)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_float)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1-beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)