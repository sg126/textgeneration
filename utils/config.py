r'''
Module to read a configuration file and
return it as a python dict for use in setting
up experiments. Also handles displaying this to
users in the terminal.
'''
# standard library imports
from __future__ import print_function, absolute_import

# third party imports
from yaml import safe_load      # safe_load is recommended (load is deprecated)
from termcolor import colored, cprint
from easydict import EasyDict as edict

# internal imports
from utils.utils import print_line


def process_config_file(config_file):
    '''
    Retrieves a yaml configuration file and processes
    it so that it can be used in other aspects of the
    project.

    Arguments
    ---------
        config_file : str
            Name of the configuration file to be parsed.
    
    Returns
    -------
        config : obj (dict)
            Configuration object (a dict)
    '''
    config = yaml_to_dict(yaml_file=config_file)
    try:
        print_line(color='green')
        cprint(f'Running: {config.experiment_name}', 'green')
        print_line(color='green')
        print('Configuration:\n')
        print_config(config)
        print_line(color='green')
        return config
    except AttributeError:
        print(
            colored('ERROR ::', 'red'),
            ' : experiment_name not specified in YAML configuration file'
        )
        exit(-1)

def yaml_to_dict(yaml_file):
    '''
    Given a yaml file path, retrieves the file and
    parses to an object that can be used in python.

    Arguments
    ---------
        yaml_file : str
            Name of file to be parsed.
    
    Returns
    -------
        config : obj (dict)
            Configuration object (a dict).
    '''
    print(yaml_file)
    with open(yaml_file, 'r') as f:
        try:
            config_dict = safe_load(f)
            config = edict(config_dict)
            return config
        except ValueError:
            print('ERROR : YAML config file formatted improperly')
            exit(-1)

def print_config(config):
    ''' Prints the configuration object in an easily readable format. '''
    for key, val in config.items():
        print(f'\t{key}  :  {val}')
