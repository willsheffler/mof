import sys, numpy as np, rpxdock as rp, os, pickle, mof
from concurrent.futures import ProcessPoolExecutor

def main():

   arg = mof.options.get_cli_args()
   arg.timer = rp.Timer().start()

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
      pdb_gen = mof.util.gen_pdbs(arg.inputs)
   else:
      # test
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!! no pdb list input, using test "only_one" !!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      # pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.103.8_0001.pdb'])
      # pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.10.3_0001.pdb'])
      pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c.2.6_0001.pdb'])
      # pdb_gen = mof.util.gen_pdbs(
      # ['/home/sheffler/debug/mof/peptides/scaffolds/C3/12res/aligned/c.10.10_0001.pdb'])
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   results = list()

   lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ = get_rotclouds(**arg)

   arg.timer.checkpoint('main')

   for pose in prepped_pdb_gen:
      mof.util.fix_bb_h_all(pose)
      for rc1, rc2 in get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):
         results.extend(
            mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **arg))
         # try:
         #    results.extend(
         #       mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **arg))
         # except Exception as e:
         #    print('some error on', rc1.amino_acid, rc2.amino_acid)
         #    print('Exception:', type(e))
         #    print(repr(e))

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

   os.makedirs(os.path.dirname(arg.output_prefix), exist_ok=True)
   for i, result in enumerate(results):
      if i in non_redundant:
         fname = arg.output_prefix + 'asym_' + result.label + '.pdb'
         print('dumping', fname)
         result.xtal_asym_pose.dump_pdb(fname)
         # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')
   arg.timer.checkpoint('dumping pdbs')

   print("DONE")

def get_rotclouds(**arg):
   arg = rp.Bunch(arg)

   chiresl_asp1 = arg.chiresl_asp1 / arg.scale_number_of_rotamers
   chiresl_asp2 = arg.chiresl_asp2 / arg.scale_number_of_rotamers
   chiresl_cys1 = arg.chiresl_cys1 / arg.scale_number_of_rotamers
   chiresl_cys2 = arg.chiresl_cys2 / arg.scale_number_of_rotamers
   chiresl_his1 = arg.chiresl_his1 / arg.scale_number_of_rotamers
   chiresl_his2 = arg.chiresl_his2 / arg.scale_number_of_rotamers
   chiresl_glu1 = arg.chiresl_glu1 / arg.scale_number_of_rotamers
   chiresl_glu2 = arg.chiresl_glu2 / arg.scale_number_of_rotamers
   chiresl_glu3 = arg.chiresl_glu3 / arg.scale_number_of_rotamers

   os.makedirs(arg.rotcloud_cache, exist_ok=True)

   params = (arg.chiresl_his1, arg.chiresl_his2, arg.chiresl_cys1, arg.chiresl_cys2,
             arg.chiresl_asp1, arg.chiresl_asp2, arg.chiresl_glu1, arg.chiresl_glu2,
             arg.chiresl_glu3, arg.maxdun_cys, arg.maxdun_asp, arg.maxdun_glu, arg.maxdun_his)
   ident = mof.util.hash_str_to_int(str(params))

   cache_file = arg.rotcloud_cache + '/%i.pickle' % ident
   if os.path.exists(cache_file):
      lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ = rp.util.load(cache_file)
   else:
      print('building rotamer clouds')
      chi_range = lambda resl: np.arange(-180, 180, resl)
      chi_asp = [chi_range(x) for x in (chiresl_asp1, chiresl_asp2)]
      chi_cys = [chi_range(x) for x in (chiresl_cys1, chiresl_cys2)]
      chi_his = [chi_range(x) for x in (chiresl_his1, chiresl_his2)]
      chi_glu = [chi_range(x) for x in (chiresl_glu1, chiresl_glu2, chiresl_glu3)]

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

      rp.util.dump([lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ], cache_file)

   return lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ

def get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):

   # return [(dC, lD)]

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
