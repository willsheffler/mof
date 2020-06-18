import sys, os, pickle, argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import mof

def read_dataset(fname):
   if not os.path.exists(fname):
      print(f'{" INPUT FILE DOESNT EXIST!!":=^80}')
      print(fname)
      sys.exit(-1)
   with open(fname, 'rb') as inp:
      info = pickle.load(inp)
      if 'xspec' in info.attrs:
         del info.attrs['xspec']
      return info

def main():
   kw = mof.options.get_cli_args()
   if not kw.inputs:
      print(f'{" NO INPUTS!":=^80}')
      print('Usage: combine_info.py <file1.pickle> <file2.pickle> ...')
      sys.exit(-1)

   # # lazy test
   # kw.inputs = [
   #    'results/mofdock__info1_kwhash4685547293031335171.pickle',
   #    'results/mofdock__info1_kwhash3015409697013192485.pickle'
   # ]

   print(f'{" reading input Datasets ":=^80}')
   info_inputs = [read_dataset(fname) for fname in tqdm(kw.inputs)]

   print(f'{" concatenatingt Datasets ":=^80}')
   info = xr.concat(info_inputs, 'result')
   ijob = np.repeat(np.arange(len(info_inputs)), [d.dims['result'] for d in info_inputs])
   info['ijob'] = (['result'], ijob)
   info.attrs.kw = [d.kw for d in info_inputs]
   if 'asym_pose_min' in info_inputs[0].attrs:
      # info.attrs.asym_pose_min = [d.asym_pose_min for d in info_inputs]
      del info.attrs['asym_pose_min']

   print(f'{" combined Dataset ":=^80}')
   print(info)

   fname = f'{kw.output_prefix}_combined_info_files.pickle'
   print(f'{" saving to ":=^80}')
   print(f'{" (use --output_prefix to change output location) ":=^80}')
   print(fname)
   os.makedirs(os.path.dirname(fname), exist_ok=True)
   with open(fname, 'wb') as out:
      pickle.dump(info, out)

   print(f'{" combine_info.py done ":=^80}')

if __name__ == '__main__':
   main()
