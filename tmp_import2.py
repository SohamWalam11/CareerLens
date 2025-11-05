import sys
import pprint
repo_root = r'e:\thisisme\CareerLens'
backend_dir = r'e:\thisisme\CareerLens\backend'
# mimic environment used in CI
sys.path.insert(0, repo_root)
sys.path.insert(0, backend_dir)
print('sys.path (top entries):')
for p in sys.path[:5]:
    print(' ', p)

try:
    import importlib
    print('\ntrying import backend.app.services.recommender')
    importlib.import_module('backend.app.services.recommender')
    print('imported backend.app.services.recommender OK')
except Exception as e:
    print('failed importing backend.app.services.recommender:', repr(e))

try:
    print('\ntrying import app.models.profile')
    importlib.import_module('app.models.profile')
    print('imported app.models.profile OK')
except Exception as e:
    print('failed importing app.models.profile:', repr(e))
    raise
