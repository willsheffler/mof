import os

def path_to_test_data(fname):
   return os.path.join(os.path.dirname(__file__), fname)
