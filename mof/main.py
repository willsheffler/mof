import sys, numpy as np, os, pickle, mof, xarray as xr, glob
import rpxdock as rp
from mof.pyrosetta_init import rosetta

# from mof.c3d2 import main_loop
from mof.c3c3 import main_loop

from mof.rotamer_cloud import get_rotclouds

# from concurrent.futures import ProcessPoolExecutor
from hashlib import sha1

def _gen_pdbs(pdblist, already_done=set()):
   for path in pdblist:
      if path not in already_done:
         yield path, rosetta.core.import_pose.pose_from_file(path)
      else:
         print(f"\n{f'!!! ALREADY COMPLETE: {path} !!!':!^80}\n")

def main_single_res():
   from mof._old_1res import main_1res
   main_1res()

def main_double_res():

   kw = mof.options.get_cli_args()
   kw.timer = rp.Timer().start()

   if len(kw.inputs) is 0:
      kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
      print(f'{"":!^80}')
      print(f'{"no pdb list input, using test only_one":!^80}')
      print(f'{str(kw.inputs):!^80}')
      print(f'{"":!^80}')
      kw.output_prefix = '_mof_main_test_output/'
      kw.spacegroups = ['i213']
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
      kw.max_score_minimized = 50.0
      kw.min_cell_size = 0
      kw.max_cell_size = 50
      kw.max_solv_frac = 0.80
      kw.debug = True
      # kw.continue_from_checkpoints = False

      kw.spacegroups = ['i213', 'p4132', 'p4332']
      # kw.spacegroups = ['p4132', 'p4332']
      if kw.debug:
         kw.spacegroups = ['i213']

      # pdb_gen = _gen_pdbs(['mof/data/peptides/c3_21res_c.103.8_0001.pdb'])
      # pdb_gen = _gen_pdbs(['mof/data/peptides/c3_21res_c.10.3_0001.pdb'])
      # pdb_gen = _gen_pdbs(
      # ['/home/sheffler/debug/mof/peptides/scaffolds/C3/12res/aligned/c.10.10_0001.pdb'])

   os.makedirs(os.path.dirname(kw.output_prefix), exist_ok=True)

   rotclouds = get_rotclouds(**kw)
   rotcloud_pairs = get_rotcloud_pairs_from_options(rotclouds, **kw)
   print('ALLOWED ROTAMER PAIRS')
   for a, b in rotcloud_pairs:
      print('   ', a.amino_acid, b.amino_acid)

   if not kw.spacegroups:
      print('\n!!!!!!!!!!!!!!!!!!!!!!!! no space groups !!!!!!!!!!!!!!!\n')
      return
   if not rotcloud_pairs:
      print('\n!!!!!!!!!!!!!!!!!!!!!!!! no rotamers allowed!!!!!!!!!!!!\n')
      return

   # parameters not to be considered as unique for checkpointing
   tohash = kw.sub(
      timer=None,
      # continue_from_checkpoints=None,
      debug=None,
      rotcloud_pairs=str([(a.amino_acid, b.amino_acid) for a, b in rotcloud_pairs]),
   )
   # # TODO: CHECKPOINTING .info file!!!!!!!
   kwhash = str(mof.util.hash_str_to_int(str(tohash)))
   # checkpoint_file = f'{kw.output_prefix}_checkpoints/{kwhash}.checkpoint'
   # os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

   already_done, ntries = set(), 1
   # if kw.continue_from_checkpoints:
   #    if os.path.exists(checkpoint_file):
   #       with open(checkpoint_file) as inp:
   #          seen = [line.strip() for line in inp]
   #          if seen: ntries = int(seen[0]) + 1
   #          already_done.update(seen[1:])
   #       if already_done:
   #          with open(checkpoint_file, 'w') as out:
   #             out.write(str(ntries) + '\n')
   #             for checkpoint in already_done:
   #                out.write(checkpoint + '\n')
   #    else:
   #       with open(checkpoint_file, 'w') as out:
   #          out.write(str(ntries) + '\n')
   pdb_gen = _gen_pdbs(kw.inputs, already_done)
   prepped_pdb_gen = mof.util.prep_poses(pdb_gen)

   # progress_total = len(kw.inputs) * len(kw.spacegroups) * len(rotcloud_pairs)
   # progress = len(already_done)

   results = list()
   kw.timer.checkpoint('main')
   success = True
   for pdbpath, pose in prepped_pdb_gen:
      print('=' * 80)
      print(f"{f'=== PDB: {pdbpath} ===':=^80}")
      print('=' * 80)
      mof.util.fix_bb_h_all(pose)

      for spacegroup in kw.spacegroups:
         print(f"  {f'= spacegroup: {spacegroup} ===':=^76}")
         search_spec = mof.xtal_search.XtalSearchSpec(
            spacegroup=spacegroup,
            pept_orig=np.array([0, 0, 0, 1]),
            pept_axis=np.array([0, 0, 1, 0]),
            # are these necessary:
            sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2'),
            ligands=['HZ3', 'DHZ3'],
            **kw,
         )
         for rc1, rc2 in rotcloud_pairs:
            # checkpoint = f'{pdbpath}_{spacegroup.replace("","_")}_{rc1.amino_acid}_{rc2.amino_acid}'
            # if checkpoint not in already_done:
            try:
               r = mof.xtal_search.xtal_search_two_residues(search_spec, pose, rc1, rc2, **kw)
               results.extend(r)
               for i, result in enumerate(results):
                  fname = kw.output_prefix + 'asym_' + result.info.label + '.pdb'
                  print('dumping', fname)
                  result.info.fname = fname
                  result.asym_pose_min.dump_pdb(fname)

            except Exception as e:
               print(f'{"SOME EXCEPTION IN RUN":=^80}')
               print(f'{f"AAs: {rc1.amino_acid} {rc2.amino_acid}":=^80}')
               print(type(e))
               print(repr(e))
               print(f'{"TRY TO DUMP PARTIAL RESULTS":=^80}')
               success = False
            # with open(checkpoint_file, 'a') as out:
            #    out.write(checkpoint + '\n')
         # with open(checkpoint_file, 'a') as out:
         #   out.write(f'{pdbpath}\n')

      if not results:
         print(kw.timer)
         print(f'{"NO RESULTS":=^80}')
         return

      kw.timer.checkpoint('main')
   if len(results) == 0:
      print('NO RESULTS')
      return

   # cluster
   scores = np.array([r.info.score for r in results])
   sorder = np.argsort(scores)  # perm to sorted
   crd = np.concatenate([r.info.bbcoords for r in results])[sorder].reshape(len(scores), -1)
   nbbatm = (results[0].asym_pose_min.size() - 1) * 3
   clustcen = rp.cluster.cookie_cutter(crd, kw.max_bb_redundancy * np.sqrt(nbbatm))
   clustcen = sorder[clustcen]
   results = [results[i] for i in clustcen]
   kw.timer.checkpoint('filter_redundancy')

   # dump pdbs
   #   for i, result in enumerate(results):
   #      fname = kw.output_prefix + 'asym_' + result.info.label + '.pdb'
   #      print('dumping', fname)
   #      result.info.fname = fname
   #      result.asym_pose_min.dump_pdb(fname)
   #      # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.info.label + '.pdb')
   kw.timer.checkpoint('dump_pdbs')

   # make Dataset
   results = mof.result.results_to_xarray(results)
   results.attrs['kw'] = kw
   rfname = f'{kw.output_prefix}_info{ntries}_kwhash{kwhash}.pickle'
   print('saving results to', rfname)
   rp.dump(results, rfname)
   print(f'{" RESULTS ":=^80}')
   print(results)
   print(f'{" END RESULTS ":=^80}')
   kw.timer.checkpoint('dump_info')

   print("main_double_res DONE")

_lblmap = dict(
   CYS='lC',
   DCYS='dC',
   ASP='lD',
   DASP='dD',
   GLU='lE',
   DGLU='dE',
   HIS='lH',
   DHIS='dH',
   HISD='lJ',
   DHISD='dJ',
)

def get_rotcloud_pairs_from_options(rotclouds, **kw):
   kw = rp.Bunch(kw)
   if len(kw.aa_pair_labels) is 1 and kw.aa_pair_labels[0].upper() == 'ALL':
      return rotcloud_pairs_all(rotclouds, **kw)
   else:
      result = list()
      pairs_listed = [_.upper() for _ in kw.aa_pair_labels]
      print('pairs_listed', pairs_listed)
      pairs = zip(pairs_listed[::2], pairs_listed[1::2])

      for lbl1, lbl2 in pairs:
         print('use AA pair', lbl1, lbl2)
         try:
            result.append((rotclouds[_lblmap[lbl1.upper()]], rotclouds[_lblmap[lbl2.upper()]]))
         except KeyError as e:
            raise ValueError('bad aa label "%s" and/or "%s"' % (lbl1, lbl2)) from e
      return result

def rotcloud_pairs_all(rotclouds, **kw):
   return [
      (rotclouds['dC'], rotclouds['dC']),  #
      (rotclouds['dC'], rotclouds['lC']),  #
      (rotclouds['lC'], rotclouds['dC']),  #
      (rotclouds['lC'], rotclouds['lC']),  #
      (rotclouds['dC'], rotclouds['dD']),
      (rotclouds['dC'], rotclouds['lD']),
      (rotclouds['lC'], rotclouds['dD']),
      (rotclouds['lC'], rotclouds['lD']),
      (rotclouds['dC'], rotclouds['dE']),
      (rotclouds['dC'], rotclouds['lE']),
      (rotclouds['lC'], rotclouds['dE']),
      (rotclouds['lC'], rotclouds['lE']),
      (rotclouds['dC'], rotclouds['dH']),
      (rotclouds['dC'], rotclouds['lH']),
      (rotclouds['lC'], rotclouds['dH']),
      (rotclouds['lC'], rotclouds['lH']),
      (rotclouds['dC'], rotclouds['dJ']),
      (rotclouds['dC'], rotclouds['lJ']),
      (rotclouds['lC'], rotclouds['dJ']),
      (rotclouds['lC'], rotclouds['lJ']),
      (rotclouds['dD'], rotclouds['dD']),  #
      (rotclouds['dD'], rotclouds['lD']),  #
      (rotclouds['lD'], rotclouds['dD']),  #
      (rotclouds['lD'], rotclouds['lD']),  #
      (rotclouds['dD'], rotclouds['dE']),  #
      (rotclouds['dD'], rotclouds['lE']),  #
      (rotclouds['lD'], rotclouds['dE']),  #
      (rotclouds['lD'], rotclouds['lE']),  #
      (rotclouds['dD'], rotclouds['dH']),
      (rotclouds['dD'], rotclouds['lH']),
      (rotclouds['lD'], rotclouds['dH']),
      (rotclouds['lD'], rotclouds['lH']),
      (rotclouds['dD'], rotclouds['dJ']),
      (rotclouds['dD'], rotclouds['lJ']),
      (rotclouds['lD'], rotclouds['dJ']),
      (rotclouds['lD'], rotclouds['lJ']),
      (rotclouds['dE'], rotclouds['dE']),  #
      (rotclouds['dE'], rotclouds['lE']),  #
      (rotclouds['lE'], rotclouds['dE']),  #
      (rotclouds['lE'], rotclouds['lE']),  #
      (rotclouds['dE'], rotclouds['dH']),
      (rotclouds['dE'], rotclouds['lH']),
      (rotclouds['lE'], rotclouds['dH']),
      (rotclouds['lE'], rotclouds['lH']),
      (rotclouds['dE'], rotclouds['dJ']),
      (rotclouds['dE'], rotclouds['lJ']),
      (rotclouds['lE'], rotclouds['dJ']),
      (rotclouds['lE'], rotclouds['lJ']),
      (rotclouds['dH'], rotclouds['dH']),  #
      (rotclouds['dH'], rotclouds['lH']),  #
      (rotclouds['lH'], rotclouds['dH']),  #
      (rotclouds['lH'], rotclouds['lH']),  #
      (rotclouds['dH'], rotclouds['dJ']),  #
      (rotclouds['dH'], rotclouds['lJ']),  #
      (rotclouds['lH'], rotclouds['dJ']),  #
      (rotclouds['lH'], rotclouds['lJ']),  #
      (rotclouds['dJ'], rotclouds['dJ']),  #
      (rotclouds['dJ'], rotclouds['lJ']),  #
      (rotclouds['lJ'], rotclouds['dJ']),  #
      (rotclouds['lJ'], rotclouds['lJ']),  #
   ]

def main():
   # import rpxdock as rp, numpy as np
   # x = rp.geom.expand_xforms([np.eye(4), np.eye(4)], 7, 50.0)
   # print(x)
   # assert 0
   # main_single_res()
   main_loop()

if __name__ == '__main__':
   main()
   print('#' * 80)
   print(f'{"main() DONE":#^80}')
   print('#' * 80)
