import pickle, os, sys
import numpy as np
import rpxdock as rp  # for clustering

_INPUTS = sys.argv[1:]

def analyze_clusters_radii(info):
   assert info.nres.min() == info.nres.max(), 'can only cluster same size pepts'
   nresults = info.sizes['result']
   nres = info.nres.data[0]
   bbcoord = info.bbcoords.data[:, :nres]
   bbcoord_flat = bbcoord.reshape(nresults, -1)
   print('cluster info:')
   print(f'radius: {0:5.3f} nunique {nresults:7,}')
   radius = 0.25
   while radius < 10:
      centers = rp.cluster.cookie_cutter(bbcoord_flat, radius)
      print(f'radius: {radius:5.3f} nunique {len(centers):7,}')
      radius *= 2

def extract_cluster_centers(info, radius):
   if radius is 0: return info
   assert info.nres.min() == info.nres.max(), 'can only cluster same size pepts'
   nresults = info.sizes['result']
   nres = info.nres.data[0]
   bbcoord = info.bbcoords.data[:, :nres]
   bbcoord_flat = bbcoord.reshape(nresults, -1)
   centers = rp.cluster.cookie_cutter(bbcoord_flat, radius)
   return info.isel(result=centers)

def most_compact_fname(info):
   return info.fname[np.argmin(info.solv_frac)].data

def get_unique_sequences(info):
   sequence_counts = [(s, x.sizes['result']) for s, x in info.groupby('sequence')]
   unique = [s for s, x in sequence_counts if x == 1]
   # print(unique)
   uinfo = info.sel(result=info.sequence.isin(unique))
   return uinfo
   # print('------------- unique sequences ----------')
   # for f in uinfo.fname:
   #    print('/home/sheffler/debug/mof/doubleres2/scaffolds_c3/9res/' + str(f.data), end='   ')

def main():
   if len(_INPUTS) is 0:
      print('MISSING INPUT!')
      sys.exit(-1)

   if len(_INPUTS) is not 1:
      print('MUST BE ONLY ONE INPUT! (Use combine_info.py to combine files)')
      sys.exit(-1)

   with open(_INPUTS[0], 'rb') as inp:
      info = pickle.load(inp)

   info = info.sortby('score')

   # analyze_clusters_radii(info)

   print('most compact xtal:')
   print(most_compact_fname(info))

   print('best score crystal')
   print(info.fname.data[0])

   print('num results by  spacegroup')
   print('total', info.sizes['result'])
   for spacegroup, x in info.groupby('spacegroup'):
      print(spacegroup, x.sizes['result'])

   print('total number of sequences', len(np.unique(info.sequence.data)))
   for radius in [0.0, 0.5, 1, 2, 4, 8, 16]:
      cinfo = extract_cluster_centers(info, radius)
      uinfo = get_unique_sequences(cinfo)
      print(f'nunique by clust radius: {radius:7.3f} unique seqs: {uinfo.sizes["result"]:5,}')

if __name__ == '__main__':
   main()
