import sys, numpy as np, rpxdock as rp, os, pickle

import mof
# from mof.xtal_search import xtal_search_single_residue, XtalSearchSpec
_DEBUG = False

def main():

   arg = rp.Bunch()
   arg.max_bb_redundancy = 1.0
   arg.err_tolerance = 1.5,
   arg.dist_err_tolerance = 1.0,
   arg.angle_err_tolerance = 15,
   arg.min_dist_to_z_axis = 6.0,
   arg.sym_axes_angle_tolerance = 5.0,
   arg.angle_to_cart_err_ratio = 20.0,
   arg.max_dun_score

   arg.chiresl_his1 = 3.0
   arg.chiresl_his2 = 8.0
   arg.chiresl_cys1 = 6.0
   arg.chiresl_cys2 = 8.0
   arg.chiresl_asp1 = 8.0
   arg.chiresl_asp2 = 5.0
   arg.chiresl_glu1 = 6.0
   arg.chiresl_glu2 = 12.0
   arg.chiresl_glu3 = 6.0

   # arg.dist_err_tolerance = 0.8
   # arg.angle_err_tolerance = 15.0
   # arg.min_dist_to_z_axis = 6.0

   # ligands = ['HZ4', 'DHZ4']
   # xspec = xtal_spec.get_xtal_spec('f432')
   # ligands = ['HZD', 'DHZD']
   # xspec = xtal_spec.get_xtal_spec(None)
   search_spec = mof.xtal_search.XtalSearchSpec(
      # spacegroup='p4132',
      spacegroup='i213',
      pept_orig=np.array([0, 0, 0, 1]),
      pept_axis=np.array([0, 0, 1, 0]),
      max_dun_score=arg.max_dun_score,
      sym_of_ligand=dict(
         HZ3='C3',
         DHZ3='C3',
         HZ4='C4',
         DHZ4='C4',
         HZD='D2',
         DHZD='D2',
         # ASP='C2',
         # CYS='C2',
         # GLU='C2',
         # HIS='C2',
      ),
      ligands=['HZ3', 'DHZ3'],
   )

   if len(sys.argv) > 1:
      pdb_gen = mof.util.gen_pdbs(sys.argv[1:])
   else:
      # test
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!! no pdb list input, using test "only_one" !!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      pdb_gen = mof.util.gen_pdbs(['mof/tests/only_one/input/c3_21res_c.107.7_0001.pdb'])
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   results = list()

   tmp_cache_file = 'tmp_cache.pickle'
   if os.path.exists(tmp_cache_file):
      rc_his, rc_cys, rc_asp, rc_glu = rp.util.load(tmp_cache_file)
   else:
      chi_range = lambda resl: np.arange(-180, 180, resl)
      chi_his = [chi_range(x) for x in (arg.chiresl_his1, arg.chiresl_his2)]
      chi_cys = [chi_range(x) for x in (arg.chiresl_cys1, arg.chiresl_cys2)]
      chi_asp = [chi_range(x) for x in (arg.chiresl_asp1, arg.chiresl_asp2)]
      chi_glu = [chi_range(x) for x in (arg.chiresl_glu1, arg.chiresl_glu2, arg.chiresl_glu3)]
      rc_his = mof.rotamer_cloud.RotCloudHisZN(grid=chi_his, max_dun_score=5.0)
      rc_cys = mof.rotamer_cloud.RotCloudCysZN(grid=chi_cys, max_dun_score=4.0)
      rc_asp = mof.rotamer_cloud.RotCloudAspZN(grid=chi_asp, max_dun_score=5.0)
      rc_glu = mof.rotamer_cloud.RotCloudGluZN(grid=chi_glu, max_dun_score=5.0)

      rp.util.dump([rc_his, rc_cys, rc_asp, rc_glu], tmp_cache_file)

   for pose in prepped_pdb_gen:
      if False:
         r = mof.xtal_search.xtal_search_single_residue(search_spec, pose, debug=_DEBUG)
      else:
         for rc1, rc2 in [
               # (rc_his, rc_his),
               # (rc_his, rc_cys),
            (rc_asp, rc_his),
               # (rc_his, rc_glu),
               # (rc_cys, rc_cys),
               # (rc_asp, rc_cys),
               # (rc_cys, rc_glu),
               # (rc_asp, rc_asp),
               # (rc_asp, rc_glu),
               # (rc_glu, rc_glu),
         ]:
            r = mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **arg)
            results.extend(r)

   if not results:
      print('---- no results ----')
      return

   xforms = np.array([r.xalign for r in results])
   non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                               max_bb_redundancy=arg.max_bb_redundancy,
                                               max_cluster=10000)
   for i, result in enumerate(results):
      if i in non_redundant:
         print('dumping', result.label)
         result.xtal_asym_pose.dump_pdb('asym_' + result.label + '.pdb')
         # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')

   print("DONE")

if __name__ == '__main__':
   main()
