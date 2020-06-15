import numpy as np, xarray as xr, pickle, os, sys

def read_dataset(fname):
   assert os.path.exists(fname)
   with open(fname, 'rb') as inp:
      ds = pickle.load(inp)
      if 'xspec' in ds.attrs:
         del ds.attrs['xspec']
      return ds

def main():
   # kw = mof.options.get_cli_args()

   inputs = [
      'results/mofdock__info1_kwhash4685547293031335171.pickle',
      'results/mofdock__info1_kwhash3015409697013192485.pickle'
   ]

   datasets = [read_dataset(fname) for fname in inputs]
   ds = xr.concat(datasets, 'result')
   ijob = np.repeat(np.arange(len(inputs)), [d.dims['result'] for d in datasets])
   ds['ijob'] = (['result'], ijob)
   ds.attrs.kw = [d.kw for d in datasets]
   if 'asym_pose_min' in datasets[0].attrs:
      ds.attrs.asym_pose_min = [d.asym_pose_min for d in datasets]

   print(ds.spacegroup)

   print('best score', ds.score.min())
   print('best score without xarray metadata', ds.score.data.min())
   print('count of results by spacegroup:')
   print(ds.spacegroup.groupby(ds.spacegroup).count())

   # with open
   # print(ds)

if __name__ == '__main__':
   main()
