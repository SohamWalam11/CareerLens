import importlib, pkgutil
import pprint
try:
    b = importlib.import_module('backend')
    print('backend package found:', b)
    print('backend.__path__ =', getattr(b, '__path__', None))
    print('\nbackend subpackages:')
    for finder, name, ispkg in pkgutil.iter_modules(b.__path__):
        print(' ', name, 'pkg?' , ispkg)
except Exception as e:
    print('failed to import backend:', repr(e))

try:
    a = importlib.import_module('backend.app')
    print('\nbackend.app found:', a)
    print('backend.app.__path__ =', getattr(a, '__path__', None))
except Exception as e:
    print('\nfailed to import backend.app:', repr(e))

try:
    print('\nTrying to import backend.app.services.recommender explicitly...')
    m = importlib.import_module('backend.app.services.recommender')
    print('OK:', m)
except Exception as e:
    print('failed to import backend.app.services.recommender:', repr(e))
