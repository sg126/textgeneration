r'''
General utility file to contain methods for tasks such as:
    - logging
    - debugging memory
    - tracking training progress
    - etc.
'''
# standard library imports
from __future__ import print_function
import os
import gc
import resource
import collections
from datetime import datetime

# third-party imports
import torch
import yaml
import gpustat
import matplotlib.pyplot as plt
from termcolor import cprint, colored
from easydict import EasyDict as edict


DATE_FORMAT = '%d-%m-%Y___%H-%M'


def debug_memory():
    '''
    Utility function to help debug tensor memory.

    Taken from https://forum.pyro.ai/t/a-clever-trick-to-debug-tensor-memory/556.
    '''
    print(f'maxrss = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                for o in gc.get_objects() if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


def show_gpu(device=1):
    '''
    Displays the current memory used by a GPU.

    Arguments
    ---------
        device : int
            Number of GPUs to display the stats for.
    '''
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))


def display_status(epoch, num_epochs, batch, num_batches,
        G_err, D_err, D_x, D_G_z1, D_G_z2):
    '''
    Outputs a string to stdout to display the status
    of training in the current batch and epoch.

    Arguments
    ---------
        epoch : int
            Current epoch number.
        num_epochs : int
            Total number of epochs.
        batch : int
            Current batch number.
        num_batches : int
            Total number of batches.
        G_err : float
            Loss for G.
        D_err : float
            Loss for D.
        D_x : float
            D's output on real data.
        D_G_z1 : float
            D's output on fake data.
        D_G_z2 : float
    '''
    D_G_z = D_G_z1 + D_G_z2
    print_line()
    print(f'[{epoch}/{num_epochs}]\n\
        [{batch}/{num_batches}]\n\
        Loss_D: {D_err}\n\
        Loss_G: {G_err}\n\
        D(X): {D_x}\t\
        D(G(z)): {D_G_z}'
    )


def display_state_dict(model=None, model_name='Model', optimizer=None):
    '''
    Outputs a string to stdout to display the
    PyTorch state_dict of the Generator, the
    Discriminator, and their respective optimizers.

    Arguments
    ---------
        model : obj             (default=None)
            The model whose state dict is to be printed.
        model_name : string     (default='Model')
            The title of the model.
        optimizer : obj         (default=None)
            The optimizer whose state dict is to be printed.
    '''
    print('================================')
    print(f'|| {model_name}\'s State Dict ||')
    print('================================')
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    print('\n\n')
    print('================================')
    print(f'|| {model_name} Optimizer State Dict ||')
    print('================================')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    print('================================')

def print_line(color=None):
    ''' Prints a line, if color is provided, then use that for line color. '''
    lines = "======================================================"
    if color is not None:
        cprint(lines, color)
    else:
        print(lines)

def save_experiment(checkpoint_path, config, seed, duration, start_time, end_time):
    '''
    Saves the results of the experiment and various other
    information regarding training to a file for future reference.

    Arguments
    ---------
        checkpoint_path : str
            Path to the final checkpoint of the model training.
        config : obj
            Configuration object used for training.
        seed : int
            Seed used in training.
        duration : str
            Duration of training.
        start_time : str
            Time of training start.
        end_time :  str
            Time of training end.
    '''
    saved_dict = dict()
    # update dict to contain contents of given config object
    saved_dict.update(config)
    
    # add all other data to the config object
    saved_dict = saved_dict.update({
        'checkpoint_path': checkpoint_path,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'used_seed': seed
    })

    # write data to YAML file in ./experiments folder
    now = datetime.now()
    filepath = f'/experiments/{config.experiment_name}_{now.strftime(DATE_FORMAT)}.yml'
    saved_filepath = filepath
    filepath = os.getcwd() + filepath
    with open(filepath, 'w') as f:
        yaml.dump(saved_dict, f, default_flow_style=False)
    
    print(f'EXPERIMENT INFORMATION SAVED TO {saved_filepath}')
    print(f'FILE CONTENTS:\n\t{print(saved_dict)}')