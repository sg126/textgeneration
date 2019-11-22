# standard library imports
import sys
from argparse import ArgumentParser


class CustomParser(ArgumentParser):
    '''
    Wrapper class for argparse.ArgumentParser
    to print out details when an error occurs
    attempting to parser CL input.
    '''

    def error(self, message):
        '''
        When an error occurs parsing from CL, write
        message to stdout and print the help statements.

        Arguments
        ---------
            message : str
                Message to write when an error occurs.
        '''
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        # "generally use sys.exit(2) for command-line syntax errors"
        # from: https://docs.python.org/3/library/sys.html#sys.exit 
        sys.exit(2)