import sys, numpy as np, rpxdock as rp

from mof import util

from mof.xtal_search import xtal_search_single_residue, XtalSearchSpec
_DEBUG = False

def main():

   # ligands = ['HZ4', 'DHZ4']
   # xspec = xtal_spec.get_xtal_spec('f432')

   # ligands = ['HZD', 'DHZD']
   # xspec = xtal_spec.get_xtal_spec(None)

   search_spec = XtalSearchSpec(
      spacegroup='p213',
      pept_orig=np.array([0, 0, 0, 1]),
      pept_axis=np.array([0, 0, 1, 0]),
      max_dun_score=4.0,
      sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2'),
      ligands=['HZ3', 'DHZ3'],
   )

   if len(sys.argv) > 1:
      pdb_gen = util.gen_pdbs(sys.argv[1])
   else:
      # test
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!! no pdb list input, using test "only_one" !!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      pdb_gen = util.gen_pdbs('mof/tests/only_one/input/only_one.list')
   prepped_pdb_gen = util.prep_poses(pdb_gen)

   results = list()

   for pose in prepped_pdb_gen:
      results.extend(xtal_search_single_residue(search_spec, pose, debug=_DEBUG))

   if not results:
      return

   xforms = np.array([r.xalign for r in results])
   non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                               max_bb_redundancy=1.0, max_cluster=10000)
   for i, result in enumerate(results):
      if i in non_redundant:
         print('dumping', result.label)
         result.xtal_asym_pose.dump_pdb('asym_' + result.label + '.pdb')
         rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')

   print("DONE")

if __name__ == '__main__':
   main()
