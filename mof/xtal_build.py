import numpy as np, rpxdock as rp

from pyrosetta import rosetta as rt
import mof
from mof.pyrosetta_init import lj_sfxn

from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as xyzMat
from mof.pyrosetta_init import make_residue
from pyrosetta import AtomID

def xtal_build(
      pdb_name,
      xspec,
      aa1,
      aa2,
      pose,
      peptide_sym,
      peptide_orig,
      peptide_axis,
      metal_sym,
      metal_origin,
      metal_sym_axis,
      rpxbody,
      tag,
      clash_dis,
      contact_dis,
      min_contacts,
      max_sym_score,
      debug=False,
      **kw,
):
   kw = rp.Bunch(kw)
   if not kw.timer: kw.timer = rp.Timer().start()

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
      # print('both sym elements not at origin')
      pass
      # raise NotImplementedError('both sym elements not at origin')

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
   ax1 = ax1 / np.linalg.norm(ax1)
   ax2 = ax2 / np.linalg.norm(ax2)

   # print(rp.homog.line_angle(metal_sym_axis, peptide_axis), np.radians(dihedral))
   assert np.allclose(rp.homog.line_angle(metal_sym_axis, peptide_axis), np.radians(dihedral),
                      atol=1e-3)
   assert np.allclose(rp.homog.line_angle(ax1, ax2), np.radians(dihedral), atol=1e-3)

   # print(rp.homog.line_angle(ax1, ax2), np.radians(dihedral), dihedral)

   # print('sym1', sym1, orig1, axis1, ax1)
   # print('sym2', sym2, orig2, axis2, ax2)

   #
   # print('== xtal_build ==')

   # Xalign, _ = rp.homog.align_lines_isect_axis2(pt1, ax1, pt2, ax2, axis1, orig1, axis2, orig2)
   Xalign, scale = rp.homog.scale_translate_lines_isect_lines(pt1, ax1, pt2, ax2, orig1, axis1,
                                                              orig2, axis2)

   xpt1, xax1 = Xalign @ pt1, Xalign @ ax1
   xpt2, xax2 = Xalign @ pt2, Xalign @ ax2
   # print('aligned1', xpt1, xax1)
   # print('aligned2', xpt2, xax2)
   assert np.allclose(rp.homog.line_angle(xax1, axis1), 0.0, atol=0.001)
   assert np.allclose(rp.homog.line_angle(xax2, axis2), 0.0, atol=0.001)

   # from previous req that ax1 isect the origin??
   # assert np.allclose(rp.homog.line_angle(xpt1, axis1), 0.0, atol=0.001)

   isect_error2 = rp.homog.line_line_distance_pa(xpt2, xax2, [0, 0, 0, 1], orig2)
   assert np.allclose(isect_error2, 0, atol=0.001)

   # isect = rp.homog.line_line_closest_points_pa(xpt2, xax2, [0, 0, 0, 1], orig2)
   # isect = (isect[0] + isect[1]) / 2
   # orig = orig2  # not always??
   # celldims = [isect[i] / o for i, o in enumerate(orig[:3]) if abs(o) > 0.001]
   celldims = scale[:3]
   assert np.allclose(min(celldims), max(celldims), atol=0.001)
   celldim = abs(min(celldims))
   if not (kw.min_cell_size <= celldim <= kw.max_cell_size):
      print('     ', xspec.spacegroup, pdb_name, aa1, aa2, 'Fail on cell_size', celldim)
      return []

   nsym = int(peptide_sym[1])
   assert pose.size() % nsym == 0
   nres_asym = pose.size() // nsym
   xtal_pose = rt.protocols.grafting.return_region(pose, 1, nres_asym)

   solv_frac = mof.filters.approx_solvent_fraction(xtal_pose, xspec, celldim)
   if kw.max_solv_frac < solv_frac:
      print('     ', xspec.spacegroup, pdb_name, aa1, aa2, 'Fail on solv_frac', solv_frac)
      return []

   # rt.numeric.xyzVec(hz[0], hz[1], hz[2]))

   xtal_pose.apply_transform_Rx_plus_v(
      xyzMat.cols(Xalign[0, 0], Xalign[1, 0], Xalign[2, 0], Xalign[0, 1], Xalign[1, 1],
                  Xalign[2, 1], Xalign[0, 2], Xalign[1, 2], Xalign[2, 2]),
      xyzVec(Xalign[0, 3], Xalign[1, 3], Xalign[2, 3]))

   # check ZN sym center location vs zn position?????????

   # print('--------------')
   # print(celldim, orig1[:3], orig2[:3])
   # print('--------------')

   if xspec.frames is None:
      # raise NotImplementedError('no sym frames for', xspec.spacegroup)
      print(f'{f" generate sym frames for {spec.spacegroup} ":}')
      dummy_scale = 100.0
      if sym1 == peptide_sym:
         redundant_point = rp.homog.hpoint(dummy_scale * orig1[:3] + 10 * axis1[:3])
      else:
         redundant_point = rp.homog.hpoint(dummy_scale * orig2[:3] + 10 * axis2[:3])
      # redundant_point = rp.homog.rand_point()
      # redundant_point[:3] *= dummy_scale

      print('redundant_point', redundant_point)
      print(Xalign @ redundant_point)

      g1 = rp.homog.hrot(axis1, (2 * np.pi / nfold1) * np.arange(nfold1), dummy_scale * orig1[:3])
      g2 = rp.homog.hrot(axis2, (2 * np.pi / nfold2) * np.arange(nfold2), dummy_scale * orig2[:3])
      g = np.concatenate([g1, g2])
      symxforms = list()
      count = 0
      for x in rp.homog.expand_xforms(g, redundant_point=redundant_point, N=12,
                                      maxrad=3.0 * dummy_scale):
         symxforms.append(x)
         print(count)
         count += 1
      # rp.dump(list(symxforms), 'i213_redundant111_n16_maxrad2.pickle')
      # symxforms = rp.load('i213_redundant111_n16_maxrad2_ORIG.pickle')
      print('num symframes', len(symxforms), type(symxforms[0]))
      symxforms = np.stack(symxforms, axis=0)
      print('symframes shape', symxforms.shape)
      symxforms[:, :3, 3] /= dummy_scale
      # print(np.around(symxforms[:, :3, 3], 3))

      rp.dump(symxforms, 'new_symframes.pickle')

      symxforms[:, :3, 3] *= celldim
      rpxbody.move_to(Xalign)
      rpxbody.dump_pdb('body_I.pdb')
      for i, x in enumerate(symxforms):
         rpxbody.move_to(x @ Xalign)
         rpxbody.dump_pdb('body_%i.pdb' % i)

      assert 0, "must rerun with newly genrated symframes"

   symxforms = xspec.frames.copy()
   symxforms[:, :3, 3] *= celldim

   # g1 = rp.homog.hrot(axis1, (2 * np.pi / nfold1) * np.arange(0, nfold1), celldim * orig1[:3])
   # g2 = rp.homog.hrot(axis2, (2 * np.pi / nfold2) * np.arange(0, nfold2), celldim * orig2[:3])
   # # # print('g1', axis1.round(3), 360 / nfold1, (celldim * orig1[:3]).round(3))
   # # # print('g2', axis2.round(3), 360 / nfold2, (celldim * orig2[:3]).round(3))
   # if swapped: g1, g2 = g2, g1
   # g = np.concatenate([g1, g2])
   # # print('swapped', swapped)
   # redundant_point = (xpt1 + xax1) if first_is_peptide else (xpt2 + xax2)
   # # redundant_point = xpt1 if first_is_peptide else xpt2
   # # print('redundancy point', redundant_point)
   # # # rpxbody.move_to(np.eye(4))
   # # # rpxbody.dump_pdb('a_body.pdb')
   # rpxbody.move_to(Xalign)
   # # # print(0, Xalign @ rp.homog.hpoint([1, 2, 3]))
   # # rpxbody.dump_pdb('a_body_xalign.pdb')

   kw.timer.checkpoint()

   clash, tot_ncontact = False, 0
   rpxbody_pdb, ir_ic = rpxbody.str_pdb(warn_on_chain_overflow=False, use_orig_coords=False)
   body_xalign = rpxbody.copy()
   body_xalign.move_to(Xalign)
   # body_xalign.dump_pdb('body_xalign.pdb')
   prev = [np.eye(4)]
   for i, x in enumerate(symxforms):
      # rp.homog.expand_xforms(g, redundant_point=redundant_point, N=6, maxrad=30)):
      if np.allclose(x, np.eye(4), atol=1e-4): assert 0
      rpxbody.move_to(x @ Xalign)
      # rpxbody.dump_pdb('clashcheck%02i.pdb' % i)
      if debug:
         pdbstr, ir_ic = rpxbody.str_pdb(start=ir_ic, warn_on_chain_overflow=False,
                                         use_orig_coords=False)
         rpxbody_pdb += pdbstr
      if np.any(rpxbody.intersect(rpxbody,
                                  np.stack(prev) @ Xalign, x @ Xalign, mindis=clash_dis)):
         clash = True
         print('     ', xspec.spacegroup, pdb_name, aa1, aa2, 'Fail on xtal clash', f'sub{i+1}')
         return []

      ncontact = rpxbody.contact_count(body_xalign, maxdis=contact_dis)
      tot_ncontact += ncontact

      prev.append(x)

      #   if clash and debug:
      #    for xprev in prev:
      #       # print(xprev.shape, (xprev @ Xalign).shape, (x @ Xalign).shape)
      #       if rpxbody.intersect(rpxbody, xprev @ Xalign, x @ Xalign, mindis=3.0):
      #          show_body_isect(rpxbody, Xalign, maxdis=3.0)
      #          rp.util.dump_str(rpxbody_pdb, 'sym_bodies.pdb')
      #    assert 0

   if tot_ncontact < min_contacts:
      print('     ', xspec.spacegroup, pdb_name, aa1, aa2, 'Fail on ncontact', tot_ncontact)
      return []

   kw.timer.checkpoint('clash_check')

   # for i, x in enumerate(symxforms):
   #    # print('sym xform %02i' % i, rp.homog.axis_ang_cen_of(x)
   #    rpxbody.move_to(x @ Xalign)
   #    rpxbody.dump_pdb('clashframe_%02i.pdb' % i)

   # assert 0

   # print('!!!!!!!!!!!!!!!! debug clash check  !!!!!!!!!!!!!!!!!!!!!!!!!')
   # rpxbody.move_to(Xalign)
   # rpxbody.dump_pdb('body.pdb')
   # symxforms = rp.homog.expand_xforms(g, redundant_point=redundant_point, N=8, maxrad=30)
   # for i, x in enumerate(symxforms):
   #    # print('sym xform %02i' % i, rp.homog.axis_ang_cen_of(x))
   #    rpxbody.move_to(x @ Xalign)
   #    rpxbody.dump_pdb('clashcheck%02i.pdb' % i)
   #    if rpxbody.intersect(rpxbody, Xalign, x @ Xalign, mindis=3.0):
   #       util.show_body_isect(rpxbody, Xalign, maxdis=3.0)
   #       rp.util.dump_str(rpxbody_pdb, 'sym_bodies.pdb')
   # assert 0

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

   nonbonded_energy = 0
   if max_sym_score < 1000:
      sympose = xtal_pose.clone()
      rt.protocols.cryst.MakeLatticeMover().apply(sympose)
      syminfo = rt.core.pose.symmetry.symmetry_info(sympose)

      # rp.util.dump_str(rpxbody_pdb, 'symbodies.pdb')
      lj_sfxn(sympose)

      nasym = len(xtal_pose.residues)
      nchain = syminfo.subunits()  # sympose.num_chains()
      bonded_subunits = []
      nterm = sympose.residue(1).xyz('N')
      cterm = sympose.residue(nasym).xyz('C')
      for i in range(1, nchain):
         nres = i * nasym + 1
         cres = i * nasym + nasym
         if 2 > cterm.distance(sympose.residue(nres).xyz('N')):
            bonded_subunits.append(i + 1)
         if 2 > nterm.distance(sympose.residue(cres).xyz('C')):
            bonded_subunits.append(i + 1)

      energy_graph = sympose.energies().energy_graph()
      eweights = sympose.energies().weights()
      nonbonded_energy = 0
      for ichain in range(1, nchain):
         if ichain + 1 in bonded_subunits: continue
         for i in range(nasym):
            ir = ichain + i + 1
            for j in range(nasym):
               jr = j + 1
               edge = energy_graph.find_edge(ir, jr)
               if not edge:
                  continue
               nonbonded_energy += edge.dot(eweights)

      # print('nonbonded_energy', nonbonded_energy)
      if nonbonded_energy > max_sym_score:
         print('     ', xspec.spacegroup, pdb_name, aa1, aa2,
               'Fail on nonbonded_energy(max_sym_score)', nonbonded_energy)
         return []

   kw.timer.checkpoint('make sympose and "nonbonded" score')

   znpos = Xalign @ metal_origin
   znres = make_residue('VZN')
   xtal_pose.append_residue_by_jump(znres, 1)
   znresi = len(xtal_pose.residues)
   znpos = xyzVec(*znpos[:3])
   zndelta = znpos - xtal_pose.residue(znresi).xyz(1)
   for ia in range(1, xtal_pose.residue(znresi).natoms() + 1):
      newxyz = zndelta + xtal_pose.residue(znresi).xyz(ia)
      # print(xtal_pose.residue(znresi).name(), ia, xtal_pose.residue(znresi).atom_name(ia), newxyz)
      xtal_pose.set_xyz(AtomID(ia, znresi), newxyz)
   # xtal_pose.dump_pdb('a_xtal_pose.pdb')
   # assert 0
   # rp.util.dump_str(rpxbody_pdb, 'a_symframes.pdb')
   # assert 0

   return [(Xalign, xtal_pose, rpxbody_pdb, tot_ncontact, nonbonded_energy, solv_frac)]
