import sys, numpy as np, rpxdock as rp, os, pickle
from concurrent.futures import ProcessPoolExecutor

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
   arg.max_dun_score = 4.0
   arg.clash_dis = 3.5
   arg.contact_dis = 7.0
   arg.min_contacts = 30
   arg.max_sym_score = 30.0
   arg.min_cell_size = 0
   arg.max_cell_size = 45

   if True:
      arg.chiresl_his1 = 3.0
      arg.chiresl_his2 = 8.0
      arg.chiresl_cys1 = 6.0
      arg.chiresl_cys2 = 8.0
      arg.chiresl_asp1 = 8.0
      arg.chiresl_asp2 = 5.0
      arg.chiresl_glu1 = 6.0
      arg.chiresl_glu2 = 12.0
      arg.chiresl_glu3 = 6.0
   else:
      arg.chiresl_his1 = 6.0
      arg.chiresl_his2 = 12.0
      arg.chiresl_cys1 = 9.0
      arg.chiresl_cys2 = 12.0
      arg.chiresl_asp1 = 12.0
      arg.chiresl_asp2 = 8.0
      arg.chiresl_glu1 = 12.0
      arg.chiresl_glu2 = 18.0
      arg.chiresl_glu3 = 12.0

   arg.timer = rp.Timer().start()

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
      **arg,
   )

   if len(sys.argv) > 1:
      pdb_gen = mof.util.gen_pdbs(sys.argv[1:])
   else:
      # test
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!! no pdb list input, using test "only_one" !!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      # pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.103.8_0001.pdb'])
      pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.101.7_0001.pdb'])
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   results = list()

   tmp_cache_file = 'tmp_cache.pickle'
   if os.path.exists(tmp_cache_file):
      lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ = rp.util.load(tmp_cache_file)
   else:
      chi_range = lambda resl: np.arange(-180, 180, resl)
      chi_asp = [chi_range(x) for x in (arg.chiresl_asp1, arg.chiresl_asp2)]
      chi_cys = [chi_range(x) for x in (arg.chiresl_cys1, arg.chiresl_cys2)]
      chi_his = [chi_range(x) for x in (arg.chiresl_his1, arg.chiresl_his2)]
      chi_glu = [chi_range(x) for x in (arg.chiresl_glu1, arg.chiresl_glu2, arg.chiresl_glu3)]

      lC = mof.rotamer_cloud.RotCloudCysZN(grid=chi_cys, max_dun_score=4.0)
      lD = mof.rotamer_cloud.RotCloudAspZN(grid=chi_asp, max_dun_score=5.0)
      lE = mof.rotamer_cloud.RotCloudGluZN(grid=chi_glu, max_dun_score=5.0)
      lH = mof.rotamer_cloud.RotCloudHisZN(grid=chi_his, max_dun_score=5.0)
      lJ = mof.rotamer_cloud.RotCloudHisdZN(grid=chi_his, max_dun_score=5.0)

      dC = mof.rotamer_cloud.RotCloudDCysZN(grid=chi_cys, max_dun_score=4.0)
      dD = mof.rotamer_cloud.RotCloudDAspZN(grid=chi_asp, max_dun_score=5.0)
      dE = mof.rotamer_cloud.RotCloudDGluZN(grid=chi_glu, max_dun_score=5.0)
      dH = mof.rotamer_cloud.RotCloudDHisZN(grid=chi_his, max_dun_score=5.0)
      dJ = mof.rotamer_cloud.RotCloudDHisdZN(grid=chi_his, max_dun_score=5.0)

      rp.util.dump([lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ], tmp_cache_file)

   arg.timer.checkpoint('main')

   for pose in prepped_pdb_gen:

      # r = mof.xtal_search.xtal_search_single_residue(search_spec, pose, debug=_DEBUG)
      # with rp.util.InProcessExecutor() as exe:
      #    # with ProcessPoolExecutor() as exe:
      #    jobs = get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ)
      #    futures = [
      #       exe.submit(mof.xtal_search.xtal_search_two_residues, search_spec, pose, rc1, rc2,
      #                  **arg) for rc1, rc2 in jobs
      #    ]
      #    results = sum([f.result() for f in futures], [])

      for rc1, rc2 in get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):
         try:
            results.extend(
               mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **arg))
         except Error as e:
            print('some error on', rc1.amino_acid, rc2.amino_acid)
            print(e)

   if not results:

      print(arg.timer)
      print('---- no results ----')
      return

   arg.timer.checkpoint('main')
   xforms = np.array([r.xalign for r in results])
   non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                               max_bb_redundancy=arg.max_bb_redundancy,
                                               max_cluster=10000)
   arg.timer.checkpoint('filter_redundancy')

   for i, result in enumerate(results):
      if i in non_redundant:
         print('dumping', result.label)
         result.xtal_asym_pose.dump_pdb('asym_' + result.label + '.pdb')
         # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')
   arg.timer.checkpoint('dumping pdbs')

   print("DONE")

def get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):

   # return [(dC, dE)]

   return [
      # (dC, dC),
      # (dC, lC),
      # (lC, dC),
      # (lC, lC),
      (dC, dD),
      (dC, lD),
      (lC, dD),
      (lC, lD),
      (dC, dE),
      (dC, lE),
      (lC, dE),
      (lC, lE),
      (dC, dH),
      (dC, lH),
      (lC, dH),
      (lC, lH),
      (dC, dJ),
      (dC, lJ),
      (lC, dJ),
      (lC, lJ),
      # (dD, dD),
      # (dD, lD),
      # (lD, dD),
      # (lD, lD),
      # (dD, dE),
      # (dD, lE),
      # (lD, dE),
      # (lD, lE),
      (dD, dH),
      (dD, lH),
      (lD, dH),
      (lD, lH),
      (dD, dJ),
      (dD, lJ),
      (lD, dJ),
      (lD, lJ),
      # (dE, dE),
      # (dE, lE),
      # (lE, dE),
      # (lE, lE),
      (dE, dH),
      (dE, lH),
      (lE, dH),
      (lE, lH),
      (dE, dJ),
      (dE, lJ),
      (lE, dJ),
      (lE, lJ),
      # (dH, dH),
      # (dH, lH),
      # (lH, dH),
      # (lH, lH),
      # (dH, dJ),
      # (dH, lJ),
      # (lH, dJ),
      # (lH, lJ),
      # (dJ, dJ),
      # (dJ, lJ),
      # (lJ, dJ),
      # (lJ, lJ),
   ]

if __name__ == '__main__':
   main()
