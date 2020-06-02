import sys, numpy as np, rpxdock as rp, os, pickle, mof, xarray as xr
from concurrent.futures import ProcessPoolExecutor

def main():

   kw = mof.options.get_cli_args()
   kw.timer = rp.Timer().start()
   kw.scale_number_of_rotamers = 0.5
   kw.max_bb_redundancy = 0.0  # 0.3
   kw.err_tolerance = 2.0
   kw.dist_err_tolerance = 1.0
   kw.angle_err_tolerance = 15
   kw.min_dist_to_z_axis = 6.0
   kw.sym_axes_angle_tolerance = 6.0
   kw.angle_to_cart_err_ratio = 20.0
   kw.max_dun_score = 6.0
   kw.clash_dis = 3.3
   kw.contact_dis = 7.0
   kw.min_contacts = 0
   kw.max_sym_score = 50.0
   kw.min_cell_size = 0
   kw.max_cell_size = 36
   kw.sample_cell_spacing = True
   kw.max_solv_frac = 0.8
   kw.debug = True

   search_spec = mof.xtal_search.XtalSearchSpec(
      spacegroup='p4132',
      # spacegroup='i213',
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
      **kw,
   )

   if len(kw.inputs) > 0:
      pdb_gen = mof.util.gen_pdbs(kw.inputs)
   else:
      fnames = ['mof/data/peptides/c.2.6_0001.pdb']
      print(f'{"":!^80}')
      print(f'{"no pdb list input, using test only_one":!^80}')
      print(f'{str(fnames):!^80}')
      print(f'{"":!^80}')

      # pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.103.8_0001.pdb'])
      # pdb_gen = mof.util.gen_pdbs(['mof/data/peptides/c3_21res_c.10.3_0001.pdb'])
      # pdb_gen = mof.util.gen_pdbs(
      # ['/home/sheffler/debug/mof/peptides/scaffolds/C3/12res/aligned/c.10.10_0001.pdb'])
      pdb_gen = mof.util.gen_pdbs(fnames)
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   results = list()

   lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ = get_rotclouds(**kw)

   kw.timer.checkpoint('main')

   try:
      for pose in prepped_pdb_gen:
         mof.util.fix_bb_h_all(pose)
         for rc1, rc2 in get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):
            # try:
            results.extend(
               mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **kw))
      success = True
      # except Exception as e:
      # print('some error on', rc1.amino_acid, rc2.amino_acid)
      # print('Exception:', type(e))
      # print(repr(e))
   except Exception as e:
      print(f'{"SOME EXCEPTION IN RUN":=^80}')
      print(e)
      print(f'{"TRY TO DUMP PARTIAL RESULTS":=^80}')
      success = False

   if not results:
      print(kw.timer)
      print(f'{"NO RESULTS":=^80}')
      return

   kw.timer.checkpoint('main')
   xforms = np.array([r.xalign for r in results])
   non_redundant = np.arange(len(results))
   if kw.max_bb_redundancy > 0.0:
      non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                                  max_bb_redundancy=kw.max_bb_redundancy,
                                                  max_cluster=10000)
   kw.timer.checkpoint('filter_redundancy')

   os.makedirs(os.path.dirname(kw.output_prefix), exist_ok=True)
   for i, result in enumerate(results):
      if i in non_redundant:
         fname = kw.output_prefix + 'asym_' + result.info.label + '.pdb.gz'
         print('dumping', fname)
         result.info.fname = fname
         result.asym_pose_min.dump_pdb(fname)
         # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.info.label + '.pdb')
   kw.timer.checkpoint('dumping pdbs')

   if len(results) == 0:
      print('NO RESULTS')
      return

   results = mof.result.results_to_xarray(results)
   results.attrs['kw'] = kw
   rfname = kw.output_prefix + 'info.pickle'
   if not success:
      rfname = kw.output_prefix + '__PARTIAL__info.pickle'
   print('saving results to', rfname)
   rp.dump(results, rfname)
   print(f'{" RESULTS ":=^80}')
   print(results)
   print(f'{" END RESULTS ":=^80}')

   # concatenate results into pandas table!!!!!!!

   print("DONE")

def get_rotclouds(**kw):
   kw = rp.Bunch(kw)

   chiresl_asp1 = kw.chiresl_asp1 / kw.scale_number_of_rotamers
   chiresl_asp2 = kw.chiresl_asp2 / kw.scale_number_of_rotamers
   chiresl_cys1 = kw.chiresl_cys1 / kw.scale_number_of_rotamers
   chiresl_cys2 = kw.chiresl_cys2 / kw.scale_number_of_rotamers
   chiresl_his1 = kw.chiresl_his1 / kw.scale_number_of_rotamers
   chiresl_his2 = kw.chiresl_his2 / kw.scale_number_of_rotamers
   chiresl_glu1 = kw.chiresl_glu1 / kw.scale_number_of_rotamers
   chiresl_glu2 = kw.chiresl_glu2 / kw.scale_number_of_rotamers
   chiresl_glu3 = kw.chiresl_glu3 / kw.scale_number_of_rotamers

   os.makedirs(kw.rotcloud_cache, exist_ok=True)

   params = (kw.chiresl_his1, kw.chiresl_his2, kw.chiresl_cys1, kw.chiresl_cys2, kw.chiresl_asp1,
             kw.chiresl_asp2, kw.chiresl_glu1, kw.chiresl_glu2, kw.chiresl_glu3, kw.maxdun_cys,
             kw.maxdun_asp, kw.maxdun_glu, kw.maxdun_his)
   ident = mof.util.hash_str_to_int(str(params))

   cache_file = kw.rotcloud_cache + '/%i.pickle' % ident
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

   return [(dD, lJ)]

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
