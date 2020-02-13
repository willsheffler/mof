import sys, pandas, numpy as np, rmsd, rpxdock as rp, rpxdock.homog as hm, xarray as xr

from mof import data, xtal_spec, util

import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec, xyzMatrix_double_t as xyzMat

_DEBUG = False

class Result:
   """mof xtal search hit"""
   def __init__(self, xspec, label, xalign, rpxbody, xtal_asym_pose, symbody_pdb):
      super(Result, self).__init__()
      self.xspec = xspec
      self.label = label
      self.xtal_asym_pose = xtal_asym_pose
      self.symbody_pdb = symbody_pdb
      self.xalign = xalign
      self.rpxbody = rpxbody

def main():

   max_dun_score = 4.0

   sym_of_ligand = dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2')

   # pyros_flags = f'-mute all -output_virtual -extra_res_fa {data.HZ3_params} -preserve_crystinfo'
   # ligands = ['HZ3', 'DHZ3']
   # xspec = xtal_spec.get_xtal_spec('p213')

   pyros_flags = f'-mute all -output_virtual -extra_res_fa {data.HZ4_params} -preserve_crystinfo'
   ligands = ['HZ4', 'DHZ4']
   xspec = xtal_spec.get_xtal_spec('f432')

   # pyros_flags = f'-mute all -output_virtual -extra_res_fa {data.HZD_params} -preserve_crystinfo'
   # ligands = ['HZD', 'DHZD']
   # xspec = xtal_spec.get_xtal_spec(None)

   pept_orig = np.array([0, 0, 0, 1])
   pept_axis = np.array([0, 0, 1, 0])

   pyrosetta.init(pyros_flags)

   pdb_gen = util.gen_pdbs(sys.argv[1])
   prepped_pdb_gen = util.prep_poses(pdb_gen)

   chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
   rts = chm.residue_type_set('fa_standard')
   scfxn = rosetta.core.scoring.ScoreFunction()
   scfxn.set_weight(rosetta.core.scoring.ScoreType.fa_dun, 1.0)

   results = list()

   for pdb in prepped_pdb_gen:
      # gets the pdb name for outputs later
      p_n = pdb.pdb_info().name().split('/')[-1]
      # gets rid of the ".pdb" at the end of the pdb name
      pdb_name = p_n[:-4]

      print(f'{pdb_name} searching')

      # check the symmetry type of the pdb
      last_res = rosetta.core.pose.chain_end_res(pdb).pop()
      total_res = int(last_res)
      sym_num = pdb.chain(last_res)
      if sym_num < 2:
         print('bad pdb', p_n)
         continue
      sym = int(sym_num)
      peptide_sym = "C%i" % sym_num

      for ires in range(1, int(total_res / sym) + 1):

         if (pdb.residue_type(ires) == rts.name_map('ALA')
             or pdb.residue_type(ires) == rts.name_map('DALA')):

            lig_poses = util.mut_to_ligand(pdb, ires, ligands, sym_of_ligand)
            bad_rots = 0
            for ilig, lig_pose in enumerate(lig_poses):
               mut_res_name, lig_sym = lig_poses[lig_pose]

               rotamers = lig_pose.residue(ires).get_rotamers()
               rotamers = util.extra_rotamers(rotamers, lb=-20, ub=21, bs=20)

               pose_num = 1
               for irot, rotamer in enumerate(rotamers):
                  #for i in range(1, len(rotamer)+1): # if I want to sample the metal-axis too
                  for i in range(len(rotamer)):
                     lig_pose.residue(ires).set_chi(i + 1, rotamer[i])
                  rot_pose = rosetta.protocols.grafting.return_region(
                     lig_pose, 1, lig_pose.size())

                  if _DEBUG:
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('1HB'), ires),
                                      xyzVec(0, 0, -2))
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('CB'), ires),
                                      xyzVec(0, 0, +0.0))
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('2HB'), ires),
                                      xyzVec(0, 0, +2))

                  scfxn(rot_pose)
                  dun_score = rot_pose.energies().residue_total_energies(ires)[
                     rosetta.core.scoring.ScoreType.fa_dun]
                  if dun_score >= max_dun_score:
                     bad_rots += 1
                     continue
                  rpxbody = rp.Body(rot_pose)

                  metal_origin = hm.hpoint(util.coord_find(rot_pose, ires, 'VZN'))
                  hz = hm.hpoint(util.coord_find(rot_pose, ires, 'HZ'))
                  ne = hm.hpoint(util.coord_find(rot_pose, ires, 'VNE'))
                  metal_his_bond = hm.hnormalized(metal_origin - ne)
                  metal_sym_axis0 = hm.hnormalized(hz - metal_origin)
                  dihedral = xspec.dihedral

                  rots_around_nezn = hm.xform_around_dof_for_vector_target_angle(
                     fix=pept_axis, mov=metal_sym_axis0, dof=metal_his_bond,
                     target_angle=np.radians(dihedral))

                  for idof, rot_around_nezn in enumerate(rots_around_nezn):
                     metal_sym_axis = rot_around_nezn @ metal_sym_axis0
                     assert np.allclose(hm.line_angle(metal_sym_axis, pept_axis),
                                        np.radians(dihedral))

                     newhz = util.coord_find(rot_pose, ires, 'VZN') + 2 * metal_sym_axis[:3]

                     aid = rosetta.core.id.AtomID(rot_pose.residue(ires).atom_index('HZ'), ires)
                     xyz = rosetta.numeric.xyzVector_double_t(newhz[0], newhz[1], newhz[2])
                     rot_pose.set_xyz(aid, xyz)

                     tag = f'{pdb_name}_{ires}_{lig_poses[lig_pose][0]}_{idof}_{pose_num}'
                     xtal_poses = make_xtal(pdb_name, xspec, rot_pose, ires, peptide_sym,
                                            pept_orig, pept_axis, lig_sym, metal_origin,
                                            metal_sym_axis, rpxbody, tag)

                     for ixtal, (xalign, xtal_pose, body_pdb) in enumerate(xtal_poses):
                        celldim = xtal_pose.pdb_info().crystinfo().A()
                        fname = f"{xspec.spacegroup.replace(' ','_')}_cell{int(celldim):03}_{tag}"
                        results.append(Result(xspec, fname, xalign, rpxbody, xtal_pose, body_pdb))

                  pose_num += 1
            # print(pdb_name, 'res', ires, 'bad rots:', bad_rots)
         else:
            continue

   if not results:
      return

   xforms = np.array([r.xalign for r in results])
   non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                               max_bb_redundancy=1.0, max_cluster=10000)
   for i, result in enumerate(results):
      if i in non_redundant:
         print('dumping', result.label)
         result.xtal_asym_pose.dump_pdb(result.label + '_asym.pdb')
         rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')

   print("DONE")

def make_xtal(pdb_name, xspec, pose, ires, peptide_sym, pept_orig, pept_axis, lig_sym,
              metal_origin, metal_sym_axis, rpxbody, tag):

   sym1 = xspec.sym1
   orig1 = xspec.orig1
   axis1 = xspec.axis1
   axis1d = xspec.axis1d

   sym2 = xspec.sym2
   orig2 = xspec.orig2
   axis2 = xspec.axis2
   axis2d = xspec.axis2d

   dihedral = xspec.dihedral

   if np.allclose(orig2, [0, 0, 0, 1]):
      sym1, sym2 = sym2, sym1
      orig1, orig2 = orig2, orig1
      axis1, axis2 = axis2, axis1
      axis1d, axis2d = axis2d, axis1d
      swapped = True  # hopefully won't need this
      assert sym1 == lig_sym
      assert sym2 == peptide_sym
      # axis2 = np.array([0.57735, 0.57735, 0.57735, 0])
      # axis1 = np.array([1, 0, 0, 0])
      # orig2 = np.array([0.5, 0.5, 0, 1])
   else:
      if not np.allclose(orig1, [0, 0, 0, 1]):
         return []
      # assert np.allclose(orig1, [0, 0, 0, 1])
      assert sym1 == peptide_sym
      assert sym2 == lig_sym
      swaped = False
      # axis1 = np.array([0.57735, 0.57735, 0.57735, 0])
      assert 0, 'maybe ok, check this new branch'

   if sym1 == peptide_sym:
      pt1, ax1 = pept_orig, pept_axis
      pt2, ax2 = metal_origin, metal_sym_axis
      first_is_peptide = True
   else:
      pt1, ax1 = metal_origin, metal_sym_axis
      pt2, ax2 = pept_orig, pept_axis
      first_is_peptide = False

   nfold1 = float(str(sym1)[1])
   nfold2 = float(str(sym2)[1])

   assert np.allclose(hm.line_angle(metal_sym_axis, pept_axis), np.radians(dihedral))
   assert np.allclose(hm.line_angle(ax1, ax2), np.radians(dihedral))

   # print(hm.line_angle(ax1, ax2), np.radians(dihedral), dihedral)

   # print('sym1', sym1, orig1, axis1, ax1)
   # print('sym2', sym2, orig2, axis2, ax2)

   Xalign, delta = hm.align_lines_isect_axis2(pt1, ax1, pt2, ax2, axis1, orig1, axis2,
                                              orig2 - orig1)
   xpt1, xax1 = Xalign @ pt1, Xalign @ ax1
   xpt2, xax2 = Xalign @ pt2, Xalign @ ax2
   # print('aligned1', xpt1, xax1)
   # print('aligned2', xpt2, xax2)
   assert np.allclose(hm.line_angle(xax1, axis1), 0.0, atol=0.001)
   assert np.allclose(hm.line_angle(xax2, axis2), 0.0, atol=0.001)
   assert np.allclose(hm.line_angle(xpt1, axis1), 0.0, atol=0.001)
   isect_error2 = hm.line_line_distance_pa(xpt2, xax2, [0, 0, 0, 1], orig2 - orig1)
   assert np.allclose(isect_error2, 0, atol=0.001)

   isect = hm.line_line_closest_points_pa(xpt2, xax2, [0, 0, 0, 1], orig2 - orig1)
   isect = (isect[0] + isect[1]) / 2
   celldims = list()
   orig = orig2  # not always??
   celldims = [isect[i] / o for i, o in enumerate(orig[:3]) if abs(o) > 0.001]
   assert np.allclose(min(celldims), max(celldims), atol=0.001)
   celldim = abs(min(celldims))

   if celldim < xspec.min_cell_size:
      return []

   print(f'{pdb_name} resi {ires:3} found xtal, celldim {celldim:7.3}')

   nsym = int(peptide_sym[1])
   assert pose.size() % nsym == 0
   nres_asym = pose.size() // nsym
   xtal_pose = rosetta.protocols.grafting.return_region(pose, 1, nres_asym)

   # hz = coord_find(xtal_pose, ires, 'VZN') + 2 * metal_sym_axis[:3]
   # xtal_pose.set_xyz(rosetta.core.id.AtomID(xtal_pose.residue(ires).atom_index('HZ'), ires),
   # rosetta.numeric.xyzVector_double_t(hz[0], hz[1], hz[2]))

   xtal_pose.apply_transform_Rx_plus_v(
      rosetta.numeric.xyzMatrix_double_t.cols(Xalign[0, 0], Xalign[1, 0], Xalign[2, 0],
                                              Xalign[0, 1], Xalign[1, 1], Xalign[2, 1],
                                              Xalign[0, 2], Xalign[1, 2], Xalign[2, 2]),
      rosetta.numeric.xyzVector_double_t(Xalign[0, 3], Xalign[1, 3], Xalign[2, 3]))

   # check ZN sym center location vs zn position

   g1 = hm.hrot(axis1, 2 * np.pi / nfold1 * np.arange(1, nfold1), celldim * orig1)
   g2 = hm.hrot(axis2, 2 * np.pi / nfold2 * np.arange(1, nfold2), celldim * orig2)
   if swapped: g1, g2 = g2, g1
   g = np.concatenate([g1, g2])
   # print('swapped', swapped)
   redundant_point = (xpt1 + xax1) if first_is_peptide else (xpt2 + xax2)
   # redundant_point += [0.0039834, 0.0060859, 0.0012353, 0]
   # print('redundancy point', redundant_point)
   # rpxbody.move_to(np.eye(4))
   # rpxbody.dump_pdb('a_body.pdb')
   rpxbody.move_to(Xalign)
   # print(0, Xalign @ hm.hpoint([1, 2, 3]))
   # rpxbody.dump_pdb('a_body_xalign.pdb')
   rpxbody_pdb, ir_ic = rpxbody.str_pdb(warn_on_chain_overflow=False, use_orig_coords=False)
   for i, x in enumerate(hm.expand_xforms(g, redundant_point=redundant_point, N=7, maxrad=50)):
      # print('sym xform', hm.axis_ang_cen_of(x))
      if np.allclose(x, np.eye(4), atol=1e-4): assert 0
      rpxbody.move_to(x @ Xalign)
      pdbstr, ir_ic = rpxbody.str_pdb(start=ir_ic, warn_on_chain_overflow=False,
                                      use_orig_coords=False)
      rpxbody_pdb += pdbstr
      # print()
      # print(i, Xalign @ hm.hpoint([1, 2, 3]))
      # print(i, x @ Xalign @ hm.hpoint([1, 2, 3]))
      # print('x.axis_angle_cen', hm.axis_angle_of(x)[1] * 180 / np.pi)

      if rpxbody.intersect(rpxbody, Xalign, x @ Xalign, mindis=3.0):
         if _DEBUG:
            show_body_isect(rpxbody, Xalign, maxdis=3.0)
            rp.util.dump_str(rpxbody_pdb, 'sym_bodies.pdb')
            assert 0
         # return []
         break  # for debugging

   # fname = f'{tag}_body_xtal.pdb'
   # print('dumping checked bodies', fname)

   ci = pyrosetta.rosetta.core.io.CrystInfo()
   ci.A(celldim)  # cell dimensions
   ci.B(celldim)
   ci.C(celldim)
   ci.alpha(90)  # cell angles
   ci.beta(90)
   ci.gamma(90)
   ci.spacegroup(xspec.spacegroup)  # sace group
   pi = pyrosetta.rosetta.core.pose.PDBInfo(xtal_pose)
   pi.set_crystinfo(ci)
   xtal_pose.pdb_info(pi)

   oldzn = hm.hpoint(util.coord_find(pose, ires, 'VZN'))
   newzn = hm.hpoint(util.coord_find(xtal_pose, ires, 'VZN'))
   assert np.allclose(newzn, Xalign @ oldzn, atol=0.001)

   # pose.dump_pdb('a_pose.pdb')
   rp.util.dump_str(rpxbody_pdb, 'a_xtal_body.pdb')
   xtal_pose.dump_pdb('a_xtal_pose.pdb')
   assert 0, 'wip: xtal pose'

   return [(Xalign, xtal_pose, rpxbody_pdb)]

if __name__ == '__main__':
   main()
