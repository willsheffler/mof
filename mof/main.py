import sys, numpy as np, rpxdock as rp, os, pickle, mof, xarray as xr, glob
from mof.pyrosetta_init import rosetta
# from concurrent.futures import ProcessPoolExecutor
from hashlib import sha1

def _gen_pdbs(pdblist, seenit=[]):
   for path in pdblist:
      path = os.path.abspath(path)
      if path not in seenit:
         yield path, rosetta.core.import_pose.pose_from_file(path)
      else:
         print(f"{f'=== ALREADY COMPLETE: {path} ===':=^80} ")

def main():

   kw = mof.options.get_cli_args()
   kw.timer = rp.Timer().start()

   if len(kw.inputs) is 0:
      kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
      print(f'{"":!^80}')
      print(f'{"no pdb list input, using test only_one":!^80}')
      print(f'{str(kw.inputs):!^80}')
      print(f'{"":!^80}')

      # kw.spacegroup = 'i213'
      kw.spacegroup = 'p4132'
      # kw.spacegroup = 'p4332'
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
      kw.max_score_minimized = 50.0
      kw.min_cell_size = 0
      kw.max_cell_size = 50
      kw.max_solv_frac = 0.8
      kw.debug = True
      kw.continue_from_checkpoints = True

      # pdb_gen = _gen_pdbs(['mof/data/peptides/c3_21res_c.103.8_0001.pdb'])
      # pdb_gen = _gen_pdbs(['mof/data/peptides/c3_21res_c.10.3_0001.pdb'])
      # pdb_gen = _gen_pdbs(
      # ['/home/sheffler/debug/mof/peptides/scaffolds/C3/12res/aligned/c.10.10_0001.pdb'])

   search_spec = mof.xtal_search.XtalSearchSpec(
      pept_orig=np.array([0, 0, 0, 1]),
      pept_axis=np.array([0, 0, 1, 0]),
      # are these necessary:
      sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2'),
      ligands=['HZ3', 'DHZ3'],
      **kw,
   )
   rotclouds = get_rotclouds(**kw)

   # parameters not to be considered as unique for checkpointing
   tohash = kw.sub(
      timer=None,
      continue_from_checkpoints=None,
      debug=None,
      rotcloud_pairs=str([(a.amino_acid, b.amino_acid) for a, b in get_jobs(**rotclouds)]),
   )

   kwhash = str(mof.util.hash_str_to_int(str(tohash)))
   checkpoint_file = f'{kw.output_prefix}_checkpoints_{kwhash}/pid{os.getpid()}.checkpoint'
   os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

   seenit, totalseen = set(), 0
   if kw.continue_from_checkpoints:
      for chk in glob.glob(f'{os.path.dirname(checkpoint_file)}/*.checkpoint'):
         with open(chk) as inp:
            pdbs_seen = [line.strip() for line in inp]
            seenit.update(pdbs_seen)
            totalseen = totalseen + 1
      if totalseen != len(seenit):
         print(f'WARNING: unique pdbs run: {len(seenit)}, total run: {totalseen}')
   pdb_gen = _gen_pdbs(kw.inputs, seenit)
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   results = list()
   kw.timer.checkpoint('main')
   success = True
   for pdbpath, pose in prepped_pdb_gen:
      print('=' * 80)
      print(f"{f'=== PDB: {pdbpath} ==='}:=^80")
      print('=' * 80)
      mof.util.fix_bb_h_all(pose)
      for rc1, rc2 in get_jobs(**rotclouds):
         # try:
         results.extend(
            mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **kw))
      # except Exception as e:
      #    print(f'{"SOME EXCEPTION IN RUN":=^80}')
      #    print(f'{f"AAs: {rc1.amino_acid} {rc2.amino_acid}":=^80}')
      #    print(type(e))
      #    print(repr(e))
      #    print(f'{"TRY TO DUMP PARTIAL RESULTS":=^80}')
      #    success = False
      with open(checkpoint_file, 'a') as out:
         out.write(f'{pdbpath}\n')

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

   return dict(lC=lC, lD=lD, lE=lE, lH=lH, lJ=lJ, dC=dC, dD=dD, dE=dE, dH=dH, dJ=dJ)

def get_jobs(lC, lD, lE, lH, lJ, dC, dD, dE, dH, dJ):

   # return [(dD, lJ)]  # I 21 3
   return [(lE, dJ)]  # P 41 3 2
   # return [(lE, dJ)]  # P 43 3 2

   return [
      (dC, dC),  # 
      (dC, lC),  # 
      (lC, dC),  # 
      (lC, lC),  # 
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
      (dD, dD),  # 
      (dD, lD),  # 
      (lD, dD),  # 
      (lD, lD),  # 
      (dD, dE),  # 
      (dD, lE),  # 
      (lD, dE),  # 
      (lD, lE),  # 
      (dD, dH),
      (dD, lH),
      (lD, dH),
      (lD, lH),
      (dD, dJ),
      (dD, lJ),
      (lD, dJ),
      (lD, lJ),
      (dE, dE),  # 
      (dE, lE),  # 
      (lE, dE),  # 
      (lE, lE),  # 
      (dE, dH),
      (dE, lH),
      (lE, dH),
      (lE, lH),
      (dE, dJ),
      (dE, lJ),
      (lE, dJ),
      (lE, lJ),
      (dH, dH),  # 
      (dH, lH),  # 
      (lH, dH),  # 
      (lH, lH),  # 
      (dH, dJ),  # 
      (dH, lJ),  # 
      (lH, dJ),  # 
      (lH, lJ),  # 
      (dJ, dJ),  # 
      (dJ, lJ),  # 
      (lJ, dJ),  # 
      (lJ, lJ),  # 
   ]

if __name__ == '__main__':
   main()
