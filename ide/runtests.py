"""
usage: python runtests.py @

this script exists for easy editor integration
"""

import sys
import os
import re
from time import perf_counter
from collections import defaultdict

_overrides = {
   #   "genrate_motif_scores.py": "PYTHONPATH=. python rpxdock/app/genrate_motif_scores.py TEST"
}

_file_mappings = {
   "xtal_search.py": ["mof/main.py"],
}
_post = defaultdict(lambda: "")

def file_has_main(file):
   with open(file) as inp:
      for l in inp:
         if l.startswith("if __name__ == "):
            return True
   return False

def testfile_of(path, bname):
   print("testfile_of", path, bname)
   t = re.sub("^mof", "mof/tests", path) + "/test_" + bname
   if os.path.exists(t):
      return t

def dispatch(file, pytest_args="--duration=5"):
   """for the love of god... clean me up"""
   file = os.path.relpath(file)
   path, bname = os.path.split(file)

   if bname in _overrides:
      oride = _overrides[bname]
      return oride, _post[bname]

   print("runtests.py dispatch", path, bname)
   if bname in _file_mappings:
      if len(_file_mappings[bname]) == 1:
         file = _file_mappings[bname][0]
         path, bname = os.path.split(file)
      else:
         assert 0

   if not file_has_main(file) and not bname.startswith("test_"):
      testfile = testfile_of(path, bname)
      if testfile:
         file = testfile
         path, bname = os.path.split(file)

   if not file_has_main(file) and bname.startswith("test_"):
      cmd = "pytest {pytest_args} {file}".format(**vars())
   elif file.endswith(".py"):
      cmd = "PYTHONPATH=. python " + file
   else:
      cmd = "pytest {pytest_args}".format(**vars())

   return cmd, _post[bname]

t = perf_counter()

post = ""
if len(sys.argv) is 1:
   cmd = "pytest"
elif len(sys.argv) is 2:
   if sys.argv[1].endswith(__file__):
      cmd = ""
   else:
      cmd, post = dispatch(sys.argv[1])
else:
   print("usage: runtests.py FILE")

print("call:", sys.argv)
print("cwd:", os.getcwd())
print("cmd:", cmd)
print(f"{' util/runtests.py running cmd in cwd ':=^80}")
sys.stdout.flush()
# if 1cmd.startswith('pytest '):
os.putenv("NUMBA_OPT", "1")
# os.putenv('NUMBA_DISABLE_JIT', '1')

# print(cmd)
os.system(cmd)

print(f"{' main command done ':=^80}")
os.system(post)
t = perf_counter() - t
print(f"{f' runtests.py done, time {t:7.3f} ':=^80}")
