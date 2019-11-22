r'''
Main module for running any agent
in the project structure.
'''
# standard library imports
from os import getcwd
from sys import version_info

# third party imports
import torch

# internal imports
from agents import *
from utils.config import process_config_file
from utils.custom_parser import CustomParser
from utils.utils import print_line


def main():
    ''' Main call to run the given config file and its associated agent. '''
    if version_info[0] < 3:
        raise Exception('ERROR: Running with Python 2, please retry with Python 3')

    parser = CustomParser()
    parser.add_argument(
        'config',
        metavar='config_yml_file',
        default='None',
        help='Name of the configuration file to use. Written in YAML.'
    )

    args = parser.parse_args()
    config_file = getcwd() + '/configs/' + args.config
    config = process_config_file(config_file)

    agent_class = globals()[config.agent]
    agent = agent_class(config=config)
    agent.run()
    agent.finalize()
    print('DONE')


if __name__ == '__main__':
    main()
