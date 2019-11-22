# standard library imports
from setuptools import setup, find_packages
from os import path
from io import open

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='textgeneration',
    version='1.0',
    description='Project to generate text',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.ccs.neu.edu/siddarthganguri/textgeneration',
    author='Siddarth Ganguri',
    author_email='ganguri.s@husky.neu.edu',
    packages=find_packages()
)