import mof, os, numpy as np, rpxdock as rp, rpxdock.homog as hm
from mof.pyrosetta_init import (rosetta, makelattice, get_sfxn, xform_pose, make_residue)
from mof.util import align_cx_pose_to_z
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta import AtomID

def main_loop():

   kw = mof.app.options_setup(get_test_kw)
   if kw.postprocess:
      return mof.app.postprocess(kw)

   rotclouds = mof.rotamer_cloud.get_rotclouds(**kw)

   pept_axis = np.array([0, 0, 1, 0])
   tetrahedral_angle = 109.47122063449069

   results = list()
   rfname = f'{kw.output_prefix}results.pickle'
   print('fname prefix:', kw.output_prefix)
   print('result fname:', rfname)

   for ipdbpath, pdbpath in enumerate(kw.inputs):

      pose = rosetta.core.import_pose.pose_from_file(pdbpath)
      if not align_cx_pose_to_z(pose, pdbpath):
         continue

      rpxbody = rp.Body(pose)

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
            sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2'),
            ligands=['HZ3', 'DHZ3'],
            **kw,
         )
         xspec = search_spec.xtal_spec
         target_angle = hm.line_angle(xspec.axis1, xspec.axis2)

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
                  lig_metal_axis = rotframe[:, 0]
                  symaxis0 = rotframe[:, 1]

                  rots_around_nezn = hm.xform_around_dof_for_vector_target_angle(
                     fix=pept_axis,
                     mov=symaxis0,
                     dof=lig_metal_axis,
                     target_angle=target_angle,
                  )

                  for ibonddof, x in enumerate(rots_around_nezn):
                     symaxis = x @ symaxis0
                     if (np.pi / 2 < hm.angle(pept_axis, symaxis)):
                        symaxis *= -1

                     assert np.allclose(hm.angle_degrees(pept_axis, symaxis),
                                        hm.angle_degrees(xspec.axis1, xspec.axis2), atol=0.01)

                     xalign = hm.align_vectors(pept_axis, symaxis, xspec.axis1, xspec.axis2)

                     # print(xalign @ pept_axis, xspec.axis1)
                     # print(xalign @ symaxis, xspec.axis2)
                     assert np.allclose(xalign @ pept_axis, xspec.axis1, atol=0.001)
                     assert np.allclose(xalign @ symaxis, xspec.axis2, atol=0.001)
                     orig_metal_pos = rotframe[:, 3]

                     xalign[:3, 3] = -(xalign @ orig_metal_pos)[:3]
                     assert np.allclose(xalign @ orig_metal_pos, [0, 0, 0, 1])
                     # xalign[:3, 3] += 5 * xspec.orig2[:3]
                     # print(xspec.orig2)
                     l = hm.hnormalized(hm.hvec(xspec.orig2))
                     l0 = hm.hpoint([0, 0, 0])
                     n = hm.hcross(xspec.axis1, xspec.orig1)
                     # p0 = xalign[:, 3]
                     p0 = xalign @ [0, 0, 0, 1]  # center of peptide
                     isect = mof.util.intersect_line_plane(p0, n, l0, l)
                     xalign[:3, 3] -= isect[:3]  # why minus? huh

                     axis1_orig = hm.line_line_closest_points_pa(
                        [0, 0, 0, 1],
                        hm.hvec(xspec.orig1),
                        xalign[:, 3],
                        xspec.axis1,
                     )
                     axis1_orig = (axis1_orig[0] + axis1_orig[1]) / 2
                     axis2_orig = xalign @ orig_metal_pos

                     for n, d in zip(axis1_orig[:3], xspec.orig1[:3]):
                        if d != 0: cell_spacing1 = n / d
                     for n, d in zip(axis2_orig[:3], xspec.orig2[:3]):
                        if d != 0: cell_spacing2 = n / d
                     cellerr = (abs(cell_spacing1 - cell_spacing2) /
                                (abs(cell_spacing1) + abs(cell_spacing2)))

                     if cell_spacing1 * cell_spacing2 < 0 or cellerr > 0.04:
                        continue
                     # cell_spacing = 0.5 * cell_spacing1 + 0.5 * cell_spacing2
                     cell_spacing = cell_spacing2  # keep d2 center intact
                     # print('cell_spacing', cell_spacing1, cell_spacing2, cellerr)

                     # what to do about negative values??
                     cell_spacing = abs(cell_spacing)
                     if abs(cell_spacing) < 10:
                        continue

                     symaxis = xalign @ symaxis

                     assert 0

                     ok1 = 0.03 > hm.line_angle(symaxisd, xspec.axis2d)
                     ok2 = 0.03 > hm.line_angle(symaxisd2, xspec.axis2d)

                     if not (ok1 or ok2):
                        continue
                     if ok2:
                        symaxisd, symaxisd2 = symaxisd2, symaxisd
                     if np.pi / 2 < hm.angle(symaxisd, xspec.axis2d):
                        symaxisd = -symaxisd

                     xyzmetal = xalign @ orig_metal_pos
                     frames1 = [
                        hm.hrot(xspec.axis1, 120, xspec.orig1 * cell_spacing),
                        hm.hrot(xspec.axis1, 240, xspec.orig1 * cell_spacing),
                        np.array(hm.hrot(symaxis, 180, xyzmetal) @ xalign),
                        np.array(hm.hrot(symaxisd, 180, xyzmetal) @ xalign),
                        np.array(hm.hrot(symaxisd2, 180, xyzmetal) @ xalign),
                     ]
                     if np.any(rpxbody.intersect(rpxbody, xalign, frames1, mindis=kw.clash_dis)):
                        continue

                     outpose0 = mof.util.mutate_one_res(
                        pose,
                        ires + 1,
                        aa,
                        rotcloud.rotchi[irot],
                        sym_num,
                     )

                     #

                     assert pose.size() % xspec.nfold1 == 0
                     nres_asym = pose.size() // xspec.nfold1
                     xtal_pose = rosetta.protocols.grafting.return_region(outpose0, 1, nres_asym)
                     mof.app.set_cell_params(xtal_pose, xspec, cell_spacing)

                     #
                     pdb_name = os.path.basename(pdbpath)

                     lattice_pose = xtal_pose.clone()
                     makelattice(lattice_pose)
                     syminfo = rosetta.core.pose.symmetry.symmetry_info(lattice_pose)
                     surfvol = rosetta.core.scoring.packing.get_surf_vol(lattice_pose, 1.4)
                     peptvol = 0.0
                     for ir, r in enumerate(lattice_pose.residues):
                        ir = ir + 1  # rosetta numbering
                        for ia in range(1, r.natoms() + 1):
                           v = surfvol.vol[AtomID(ia, ir)]
                           if not np.isnan(v): peptvol += v
                     peptvol /= syminfo.subunits()
                     peptvol *= xspec.nsubs
                     print(f'peptvol {peptvol}')

                     # solv_frac = mof.filters.approx_solvent_fraction(xtal_pose, xspec, cell_spacing)
                     solv_frac = max(0.0, 1.0 - peptvol / cell_spacing**3)
                     if kw.max_solv_frac < solv_frac:
                        print('     ', xspec.spacegroup, pdb_name, aa, 'Fail on solv_frac',
                              solv_frac)
                        continue
                     else:
                        print('     ', xspec.spacegroup, pdb_name, aa, 'Win  on solv_frac',
                              solv_frac)

                     #

                     xform_pose(xtal_pose, xalign)
                     mof.app.addZN(xtal_pose, xyzmetal)

                     sfxn_min = get_sfxn('minimize')
                     xtal_pose_min, mininfo = mof.app.minimize_oneres(
                        sfxn_min, xspec, xtal_pose, **kw)
                     if xtal_pose_min is None:
                        print(f'min failed, xtal_pose_min is None')
                        continue

                     print(f'{len(results):5} {iaa:3} {ires:3} {irot:5} {ibonddof:2} ang axisd',
                           f'{hm.angle_degrees(symaxisd, xspec.axis2d):7.3}',
                           f'{hm.angle_degrees(symaxisd2, xspec.axis2d):7.3}',
                           f'{cell_spacing:7.3}', f'farep {mininfo.score_fa_rep:7.3}',
                           f'solv {solv_frac:5.3}')

                     if mininfo.score_fa_rep > kw.max_score_minimized / 3 * nresasym:
                        # if mininfo.score_wo_cst > kw.max_score_minimized:
                        print(f'score_fa_rep fail {mininfo.score_fa_rep}')
                        continue

                     tag = ''.join([
                        f'_solv{int(solv_frac*100):02}_',
                        f'{os.path.basename(pdbpath)}',
                        f'_nres{nresasym}',
                        f'_cell{int(cell_spacing):03}_',
                        f'{aa}_',
                        f'{len(results):06}',
                     ])
                     fn = kw.output_prefix + tag + '.pdb'

                     result = mof.app.prepare_result(**vars(), **kw)
                     result.pdb_fname = fn
                     result.spacegroup = spacegroup
                     result.aa = aa
                     result.seq = xtal_pose_min.sequence()
                     result.ires = ires
                     result.irot = irot
                     result.ibonddof = ibonddof
                     results.append(result)

                     print(f'{f" DUMP PDB {fn} ":!^100}')
                     rp.dump(result, fn[:-4] + '.pickle')
                     xtal_pose_min.dump_pdb(fn)

   if not results:
      print(f'{"":!^100}')
      print('NO RESULTS!!!')
      print(f'{"":!^100}')
      print('DONE')

   return

def get_test_kw(kw):
   if not kw.inputs:
      kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
      print(f'{"":!^80}')
      print(f'{"no pdb list input, using test only_one":!^80}')
      print(f'{str(kw.inputs):!^80}')
      print(f'{"":!^80}')

   kw.spacegroups = ['p213']
   kw.output_prefix = '_mof_main_test_output' + '_'.join(kw.spacegroups) + '/'
   kw.scale_number_of_rotamers = 0.5
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
