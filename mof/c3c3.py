import mof, os, numpy as np, rpxdock as rp, rpxdock.homog as hm
from mof.pyrosetta_init import (rosetta, makelattice, get_sfxn, xform_pose, make_residue)
from mof.util import align_cx_pose_to_z, variant_remove
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta import AtomID, get_score_function

def main_loop():

   kw = mof.app.options_setup(get_test_kw, verbose=False)

   if kw.postprocess:
      return mof.app.postprocess(kw)

   pept_axis = np.array([0, 0, 1, 0])
   pept_orig = np.array([0, 0, 0, 1])

   results = list()
   rfname = f'{kw.output_prefix}results.pickle'
   print('fname prefix:', kw.output_prefix)
   print('result fname:', rfname)

   sfxn = get_score_function()

   results = list()

   for ipdbpath, pdbpath in enumerate(kw.inputs):

      pose = rosetta.core.import_pose.pose_from_file(pdbpath)
      if not align_cx_pose_to_z(pose, pdbpath):
         print(f"WARNING failed align_cx_pose_to_z: {pdbpath}")
      variant_remove(pose)
      rpxbody = rp.Body(pose)

      rotclouds = mof.rotamer_cloud.get_rotclouds(**kw)

      # pose.dump_pdb('rotcloud_all_super_scaffold.pdb')
      # append = False
      # for ires in (1, 2):
      #    for k, v in rotclouds.items():
      #       if k != 'lB': continue
      #       v.dump_pdb('rotcloud_all_super.pdb', append=append, position=rpxbody.stub[ires])
      #       append = True

      minscore = 9e9

      print()
      print(f'{" %i of %i "%(ipdbpath+1 , len(kw.inputs)):#^80}')
      print(pdbpath)
      print(f'{"":#^80}')

      for spacegroup in kw.spacegroups:

         search_spec = mof.xtal_search.XtalSearchSpec(
            spacegroup=spacegroup,
            pept_orig=np.array([0, 0, 0, 1]),
            pept_axis=np.array([0, 0, 1, 0]),
            # are these necessary:
            sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2',
                               BPY='C3'),
            ligands=['HZ3', 'DHZ3'],
            **kw,
         )
         xspec = search_spec.xtal_spec
         target_angle = hm.line_angle_degrees(xspec.axis1, xspec.axis2)

         # sym_num = pose.chain(len(pose.residues))
         sym_num = xspec.nfold1
         nresasym = len(pose.residues) // sym_num
         # print(sym_num, nresasym)
         # assert 0

         for iaa, aa in enumerate(kw.aa_labels):

            rotcloud = rotclouds[mof.app.lblmap[aa]]

            for ires in range(nresasym):

               print(f'{f" LOOP {spacegroup} {aa} {ires} ":*^80}')
               stub = rpxbody.stub[ires]

               rotframes = stub @ rotcloud.rotframes

               for irot, rotframe in enumerate(rotframes):
                  ligsymaxis = rotframe[:, 2]  # z axis

                  if np.pi / 2 < hm.angle(pept_axis, ligsymaxis):
                     ligsymaxis *= -1

                  angle = hm.angle_degrees(pept_axis, ligsymaxis)
                  if abs(angle - target_angle) > kw.angle_err_tolerance:
                     continue

                  # print('foo', target_angle, angle, kw.angle_err_tolerance, ligsymaxis[:3],
                  # rotcloud.rotchi[irot])

                  orig_metal_pos = rotframe[:, 3]

                  # print('bar', hm.angle_degrees(xspec.axis1, xspec.axis2))
                  # print('baz', hm.angle_degrees(pept_axis, ligsymaxis))
                  # print('bla', ligsymaxis)

                  # xalign = hm.align_vectors(pept_axis, ligsymaxis, xspec.axis1, xspec.axis2)

                  # xalign[:3, 3] = -(xalign @ orig_metal_pos)[:3]

                  xalign, delta = hm.align_lines_isect_axis2(
                     pept_orig, pept_axis, orig_metal_pos, ligsymaxis, xspec.axis1, xspec.orig1,
                     xspec.axis2, xspec.orig2 - xspec.orig1, strict=False)

                  # print(xalign @ pept_axis, xspec.axis1)
                  # print(xalign @ ligsymaxis, xspec.axis2)
                  aligned_pept_axis = xalign @ pept_axis
                  aligned_ligsym_axis = xalign @ ligsymaxis
                  aligned_metal_pos = xalign @ orig_metal_pos
                  assert np.allclose(aligned_pept_axis, xspec.axis1, atol=kw.angle_err_tolerance)
                  assert np.allclose(aligned_ligsym_axis, xspec.axis2,
                                     atol=kw.angle_err_tolerance)

                  _, isect = hm.line_line_closest_points_pa(aligned_metal_pos,
                                                            aligned_ligsym_axis, [0, 0, 0, 1],
                                                            xspec.orig2 - xspec.orig1)
                  cell_spacing = abs(isect[2] / xspec.orig2[2])
                  if cell_spacing < 10 or cell_spacing > 25:
                     continue
                  # print('cell_spacing', cell_spacing)

                  # xalign[:3, 3] += 5 * xspec.orig2[:3]
                  # print(xspec.orig2)

                  outpose0 = mof.util.mutate_one_res(pose, ires + 1, aa, rotcloud.rotchi[irot],
                                                     sym_num)
                  xform_pose(outpose0, xalign)
                  outasym = rosetta.protocols.grafting.return_region(outpose0, 1, nresasym)
                  ci = rosetta.core.io.CrystInfo()
                  ci.A(cell_spacing)  # cell dimensions
                  ci.B(cell_spacing)
                  ci.C(cell_spacing)
                  ci.alpha(90)  # cell angles n
                  ci.beta(90)
                  ci.gamma(90)
                  ci.spacegroup(xspec.spacegroup)  # sace group
                  pi = rosetta.core.pose.PDBInfo(outasym)
                  pi.set_crystinfo(ci)
                  outasym.pdb_info(pi)

                  sympose = outasym.clone()
                  rosetta.protocols.cryst.MakeLatticeMover().apply(sympose)
                  conf = sympose.conformation().clone()
                  assert rosetta.core.conformation.symmetry.is_symmetric(conf)
                  syminfo = rosetta.core.pose.symmetry.symmetry_info(sympose)

                  nasym = outasym.size()
                  conf.declare_chemical_bond(nasym, 'C', 1, 'N')
                  sympose.set_new_conformation(conf)
                  sympose.set_new_energies_object(
                     rosetta.core.scoring.symmetry.SymmetricEnergies())

                  sc = sfxn.score(sympose)
                  # print('score', sc)
                  minscore = min(minscore, sc)
                  if sc > 7000:
                     continue

                  syminfo = rosetta.core.pose.symmetry.symmetry_info(sympose)
                  surfvol = rosetta.core.scoring.packing.get_surf_vol(sympose, 1.4)
                  peptvol = 0.0
                  for ir in range(1, syminfo.get_nres_subunit() + 1):
                     res = sympose.residue(ir)
                     ir = ir + 1  # rosetta numbering
                     for ia in range(1, res.natoms() + 1):
                        v = surfvol.vol[AtomID(ia, ir)]
                        if not np.isnan(v): peptvol += v
                  # peptvol /= syminfo.subunits()
                  peptvol *= xspec.nsubs
                  print(f'peptvol {peptvol} {peptvol / cell_spacing**3}')
                  solv_frac = max(0.0, 1.0 - peptvol / cell_spacing**3)

                  tag = ''.join([
                     f'_cell{int(cell_spacing):03}_',
                     f'_sc{int(sc):05}_',
                     f'_solv{int(solv_frac*100):02}_',
                     f'{os.path.basename(pdbpath)}',
                     f'_nres{nresasym}',
                     f'{aa}_',
                     f'{len(results):06}',
                  ])
                  fn = kw.output_prefix + tag + '.pdb'

                  print('HIT %7.3f' % sc, outasym)
                  results.append([sc, cell_spacing, sympose])
                  if True:
                     outasym.dump_pdb(fn)
                     # outpose0.dump_pdb('xalign.pdb')
                     # xform_pose(outpose0,
                     # hm.hrot(xalign @ ligsymaxis, 120, xalign @ orig_metal_pos))
                     # outpose0.dump_pdb('xalign2.pdb')
                     # xform_pose(outpose0,
                     # hm.hrot(xalign @ ligsymaxis, 120, xalign @ orig_metal_pos))
                     # outpose0.dump_pdb('xalign3.pdb')
                     # assert 0

                  # get cell cell_spacing
                  # dump cryst1 pdb
                  # clash check
      print('minscore', minscore)

   results.sort

   if not results:
      print(f'{"":!^100}')
      print('NO RESULTS!!!')
      print(f'{"":!^100}')
      print('DONE')

   return results

validated_c3 = [
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/PDD-xtal-structure.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/Baby-xtal-structure.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.100.2_0010_LETO.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.10.8_0004_DERRICK.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.1.4_0004_LONDRILUN.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.1.7_0006_0001_0001.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.18.7_0007_0001_0001_NIDORAN.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.20.7_0002_DARLA.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.25.7_0005_0001_0001_UTRE.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.2.7_0003_0001_0001_EKANS.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.28.8_0005_0001_0001_POULUNO.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.34.3_0008_0001_VENI.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.43.10_0002_0001_0001.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.60.1_0002_0001_JEAN.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.6.4_0009_MELANGE.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.80.1_0003_VECHE.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/c.94.1_0008_0001_DEWEY.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/naep_c.11.10_0002_0001_0001_BLACK.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/naep_c.18.6_0009_0001_0001_0001.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/naep_c.42.3_0008_0001_0001_0001.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/naep_c.71.1_0002_0001_0001_0001.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/pd_c.59.2_0005_0001_0001_CATERPIE.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/result00008_0000883_0_0003_SHISO.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/Sage-xtal-structure.pdb',
   '/home/sheffler/debug/mof/peptides/NMR_Xtal_validated_designs/C3/Sporty-xtal-structure.pdb',
]

def get_test_kw(kw):
   if not kw.inputs:
      kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
      # kw.inputs = validated_c3
      print(f'{"":!^80}')
      print(f'{"no pdb list input, using test only_one":!^80}')
      print(f'{str(kw.inputs):!^80}')
      print(f'{"":!^80}')

   kw.spacegroups = ['p213']
   kw.aa_labels = ['BPY']
   kw.output_prefix = '_mof_test_c3c3' + '_'.join(kw.spacegroups) + '/'
   kw.angle_err_tolerance = 3
   kw.scale_number_of_rotamers = 0.25
   kw.max_bb_redundancy = 2.0
   kw.max_dun_score = 4.0
   kw.clash_dis = 3.3
   kw.contact_dis = 7.0
   kw.min_contacts = 0
   kw.max_score_minimized = 40.0
   kw.min_cell_size = 0
   kw.max_cell_size = 50
   kw.max_solv_frac = 0.80
   kw.debug = True
   # kw.continue_from_checkpoints = False
   return kw
