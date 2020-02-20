import os

def test_data_path(fname):
   return os.path.join(os.path.dirname(__file__), fname)
