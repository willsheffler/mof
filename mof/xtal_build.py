import numpy as np, rpxdock as rp

from pyrosetta import rosetta as rt
from mof import util

from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as xyzMat

def xtal_build(
      pdb_name,
      xspec,
      pose,
      peptide_sym,
      peptide_orig,
      peptide_axis,
      metal_sym,
      metal_origin,
      metal_sym_axis,
      rpxbody,
      tag,
      debug=False,
):

   sym1 = xspec.sym1
   orig1 = xspec.orig1
   axis1 = xspec.axis1
   axis1d = xspec.axis1d

   sym2 = xspec.sym2
   orig2 = xspec.orig2
   axis2 = xspec.axis2
   axis2d = xspec.axis2d
   hack_offset = np.array([0, 0, 0, 0])

   dihedral = xspec.dihedral

   if np.allclose(orig2, [0, 0, 0, 1]):
      sym1, sym2 = sym2, sym1
      orig1, orig2 = orig2, orig1
      axis1, axis2 = axis2, axis1
      axis1d, axis2d = axis2d, axis1d
      swapped = True  # hopefully won't need this
      assert sym1 == metal_sym
      assert sym2 == peptide_sym
      # axis2 = np.array([0.57735, 0.57735, 0.57735, 0])
      # axis1 = np.array([1, 0, 0, 0])
      # orig2 = np.array([0.5, 0.5, 0, 1])
   elif np.allclose(orig1, [0, 0, 0, 1]):
      # assert np.allclose(orig1, [0, 0, 0, 1])
      assert sym1 == peptide_sym
      assert sym2 == metal_sym
      swapped = False
      # axis1 = np.array([0.57735, 0.57735, 0.57735, 0])
      # assert 0, 'maybe ok, check this new branch'
   else:
      swapped = False
      hack_offset = rp.homog.hvec(orig1)
      orig1 = orig1 - hack_offset
      orig2 = orig2 - hack_offset
      assert sym1 == peptide_sym
      assert sym2 == metal_sym
      raise NotImplementedError('both sym elements not at origin')

   if sym1 == peptide_sym:
      pt1, ax1 = peptide_orig, peptide_axis
      pt2, ax2 = metal_origin, metal_sym_axis
      first_is_peptide = True
   else:
      pt1, ax1 = metal_origin, metal_sym_axis
      pt2, ax2 = peptide_orig, peptide_axis
      first_is_peptide = False

   nfold1 = float(str(sym1)[1])
   nfold2 = float(str(sym2)[1])

   # print(rp.homog.line_angle(metal_sym_axis, peptide_axis), np.radians(dihedral))
   assert np.allclose(rp.homog.line_angle(metal_sym_axis, peptide_axis), np.radians(dihedral))
   assert np.allclose(rp.homog.line_angle(ax1, ax2), np.radians(dihedral))

   # print(rp.homog.line_angle(ax1, ax2), np.radians(dihedral), dihedral)

   # print('sym1', sym1, orig1, axis1, ax1)
   # print('sym2', sym2, orig2, axis2, ax2)

   Xalign, delta = rp.homog.align_lines_isect_axis2(pt1, ax1, pt2, ax2, axis1, orig1, axis2,
                                                    orig2 - orig1)
   xpt1, xax1 = Xalign @ pt1, Xalign @ ax1
   xpt2, xax2 = Xalign @ pt2, Xalign @ ax2
   # print('aligned1', xpt1, xax1)
   # print('aligned2', xpt2, xax2)
   assert np.allclose(rp.homog.line_angle(xax1, axis1), 0.0, atol=0.001)
   assert np.allclose(rp.homog.line_angle(xax2, axis2), 0.0, atol=0.001)
   assert np.allclose(rp.homog.line_angle(xpt1, axis1), 0.0, atol=0.001)
   isect_error2 = rp.homog.line_line_distance_pa(xpt2, xax2, [0, 0, 0, 1], orig2 - orig1)
   assert np.allclose(isect_error2, 0, atol=0.001)

   isect = rp.homog.line_line_closest_points_pa(xpt2, xax2, [0, 0, 0, 1], orig2 - orig1)
   isect = (isect[0] + isect[1]) / 2
   celldims = list()
   orig = orig2  # not always??
   celldims = [isect[i] / o for i, o in enumerate(orig[:3]) if abs(o) > 0.001]
   assert np.allclose(min(celldims), max(celldims), atol=0.001)
   celldim = abs(min(celldims))

   if celldim < xspec.min_cell_size:
      return []

   nsym = int(peptide_sym[1])
   assert pose.size() % nsym == 0
   nres_asym = pose.size() // nsym
   xtal_pose = rt.protocols.grafting.return_region(pose, 1, nres_asym)

   # rt.numeric.xyzVec(hz[0], hz[1], hz[2]))

   xtal_pose.apply_transform_Rx_plus_v(
      xyzMat.cols(Xalign[0, 0], Xalign[1, 0], Xalign[2, 0], Xalign[0, 1], Xalign[1, 1],
                  Xalign[2, 1], Xalign[0, 2], Xalign[1, 2], Xalign[2, 2]),
      xyzVec(Xalign[0, 3], Xalign[1, 3], Xalign[2, 3]))

   # check ZN sym center location vs zn position

   g1 = rp.homog.hrot(axis1, 2 * np.pi / nfold1 * np.arange(1, nfold1), celldim * orig1[:3])
   g2 = rp.homog.hrot(axis2, 2 * np.pi / nfold2 * np.arange(1, nfold2), celldim * orig2[:3])
   # print('g1', axis1.round(3), 360 / nfold1, (celldim * orig1[:3]).round(3))
   # print('g2', axis2.round(3), 360 / nfold2, (celldim * orig2[:3]).round(3))
   if swapped: g1, g2 = g2, g1
   g = np.concatenate([g1, g2])
   # print('swapped', swapped)
   redundant_point = (xpt1 + xax1) if first_is_peptide else (xpt2 + xax2)
   # redundant_point += [0.0039834, 0.0060859, 0.0012353, 0]
   # print('redundancy point', redundant_point)
   # rpxbody.move_to(np.eye(4))
   # rpxbody.dump_pdb('a_body.pdb')
   rpxbody.move_to(Xalign)
   # print(0, Xalign @ rp.homog.hpoint([1, 2, 3]))
   # rpxbody.dump_pdb('a_body_xalign.pdb')

   rpxbody_pdb, ir_ic = rpxbody.str_pdb(warn_on_chain_overflow=False, use_orig_coords=False)
   for i, x in enumerate(
         rp.homog.expand_xforms(g, redundant_point=redundant_point, N=7, maxrad=50)):
      # print('sym xform %02i' % i, rp.homog.axis_ang_cen_of(x))
      if np.allclose(x, np.eye(4), atol=1e-4): assert 0
      rpxbody.move_to(x @ Xalign)
      # rpxbody.dump_pdb('clashcheck%02i.pdb' % i)
      pdbstr, ir_ic = rpxbody.str_pdb(start=ir_ic, warn_on_chain_overflow=False,
                                      use_orig_coords=False)
      rpxbody_pdb += pdbstr
      # print()
      # print(i, Xalign @ rp.homog.hpoint([1, 2, 3]))
      # print(i, x @ Xalign @ rp.homog.hpoint([1, 2, 3]))
      # print('x.axis_angle_cen', rp.homog.axis_angle_of(x)[1] * 180 / np.pi)

      if rpxbody.intersect(rpxbody, Xalign, x @ Xalign, mindis=3.0):
         if debug:
            show_body_isect(rpxbody, Xalign, maxdis=3.0)
            rp.util.dump_str(rpxbody_pdb, 'sym_bodies.pdb')
            assert 0
         # break  # for debugging
         return []

   # assert 0, 'xtal build after clashcheck'

   # fname = f'{tag}_body_xtal.pdb'
   # print('dumping checked bodies', fname)

   ci = rt.core.io.CrystInfo()
   ci.A(celldim)  # cell dimensions
   ci.B(celldim)
   ci.C(celldim)
   ci.alpha(90)  # cell angles
   ci.beta(90)
   ci.gamma(90)
   ci.spacegroup(xspec.spacegroup)  # sace group
   pi = rt.core.pose.PDBInfo(xtal_pose)
   pi.set_crystinfo(ci)
   xtal_pose.pdb_info(pi)

   # pose.dump_pdb('a_pose.pdb')
   # rp.util.dump_str(rpxbody_pdb, 'a_xtal_body.pdb')
   # xtal_pose.dump_pdb('a_xtal_pose.pdb')
   # assert 0, 'wip: xtal pose'

   return [(Xalign, xtal_pose, rpxbody_pdb)]
