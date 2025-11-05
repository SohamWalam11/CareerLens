import sys, pprint
pprint.pprint(sys.path)
try:
    import importlib
    m = importlib.import_module('backend.app.services')
    print('import succeeded:', m)
except Exception as e:
    print('import failed:', repr(e))
    raise
