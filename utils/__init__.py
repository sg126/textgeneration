r'''
'''
# standard library imports
from os import path, listdir
from sys import modules


path = path.dirname(path.abspath(__file__))


for py in [f[:-3] for f in listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for c in classes:
        setattr(modules[__name__], c.__name__, c)