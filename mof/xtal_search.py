import mof, rpxdock as rp, numpy as np
from rpxdock import homog as hm

import mof.pyrosetta_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as rVec

from pyrosetta import rosetta as rt, init as pyrosetta_init

class XtalSearchSpec(object):
   """stuff needed for pepdide xtal search"""
   def __init__(self, spacegroup, pept_axis, pept_orig, ligands, sym_of_ligand, max_dun_score,
                **arg):
      super(XtalSearchSpec, self).__init__()
      self.spacegroup = spacegroup
      self.pept_axis = pept_axis
      self.pept_orig = pept_orig
      self.ligands = ligands
      self.sym_of_ligand = sym_of_ligand
      self.max_dun_score = max_dun_score
      self.xtal_spec = mof.xtal_spec.get_xtal_spec(self.spacegroup)
      self.chm = rt.core.chemical.ChemicalManager.get_instance()
      self.rts = self.chm.residue_type_set('fa_standard')
      self.dun_sfxn = rt.core.scoring.ScoreFunction()
      self.dun_sfxn.set_weight(rt.core.scoring.ScoreType.fa_dun, 1.0)
      self.sfxn_filter = rt.core.scoring.ScoreFunction()
      self.sfxn_filter.set_weight(rt.core.scoring.ScoreType.fa_atr, 1.00)
      self.sfxn_filter.set_weight(rt.core.scoring.ScoreType.fa_rep, 0.55)
      # self.sfxn_filter = rt.core.scoring.get_score_function()

def xtal_search_two_residues(
      search_spec,
      pose,
      rotcloud1base,
      rotcloud2base,
      err_tolerance,
      dist_err_tolerance,
      angle_err_tolerance,
      min_dist_to_z_axis,
      sym_axes_angle_tolerance,
      angle_to_cart_err_ratio,
      **arg,
):
   arg = rp.Bunch(arg)
   if not arg.timer: arg.timer = rp.Timer().start()
   arg.timer.checkpoint()

   spec = search_spec
   xspec = spec.xtal_spec

   results = list()

   dont_replace_these_aas = [spec.rts.name_map('CYS'), spec.rts.name_map('PRO')]

   farep_orig = search_spec.sfxn_filter(pose)

   p_n = pose.pdb_info().name().split('/')[-1]
   # gets rid of the ".pdb" at the end of the pdb name
   pdb_name = p_n[:-4]

   print(f'{pdb_name} searching', rotcloud1base.amino_acid, rotcloud2base.amino_acid)

   # check the symmetry type of the pdb
   last_res = rt.core.pose.chain_end_res(pose).pop()
   total_res = int(last_res)

   sym_num = 3  # pose.chain(last_res)
   sym = 3  # int(sym_num)
   if sym_num < 2:
      print('bad pdb', p_n)
      return list()
   asym_nres = int(total_res / sym)
   peptide_sym = "C%i" % sym_num

   rpxbody = rp.Body(pose)

   for ires1 in range(1, asym_nres + 1):
      # if pose.residue_type(ires1) not in (spec.rts.name_map('ALA'), spec.rts.name_map('DALA')):
      if pose.residue_type(ires1) in dont_replace_these_aas: continue
      stub1 = rpxbody.stub[ires1 - 1]

      arg.timer.checkpoint('xtal_search')
      rots1ok = min_dist_to_z_axis < np.linalg.norm(
         (stub1 @ rotcloud1base.rotframes)[:, :2, 3], axis=1)
      if 0 == np.sum(rots1ok): continue
      rotcloud1 = rotcloud1base.subset(rots1ok)
      rotframes1 = stub1 @ rotcloud1.rotframes
      # rotcloud1.dump_pdb(f'cloud_a_{ires1:02}.pdb', position=stub1)

      arg.timer.checkpoint('position rotcloud')

      range2 = range(1, int(total_res) + 1)
      if rotcloud1base is rotcloud2base: range2 = range(ires1 + 1, int(total_res) + 1)
      for ires2 in range2:
         if ires1 == ((ires2 - 1) % asym_nres + 1): continue
         if pose.residue_type(ires2) in dont_replace_these_aas:
            continue
         stub2 = rpxbody.stub[ires2 - 1]
         # rotcloud2.dump_pdb(f'cloud_b_{ires2:02}.pdb', position=stub2)

         arg.timer.checkpoint('xtal_search')
         rots2ok = min_dist_to_z_axis < np.linalg.norm(
            (stub2 @ rotcloud2base.rotframes)[:, :2, 3], axis=1)
         if 0 == np.sum(rots2ok): continue
         rotcloud2 = rotcloud2base.subset(rots2ok)
         rotframes2 = stub2 @ rotcloud2.rotframes

         arg.timer.checkpoint('rotcloud positioning')

         dist = rotframes1[:, :, 3].reshape(-1, 1, 4) - rotframes2[:, :, 3].reshape(1, -1, 4)
         dist = np.linalg.norm(dist, axis=2)

         arg.timer.checkpoint('rotcloud dist')

         # if np.min(dist) < 1.0:
         # print(f'{ires1:02} {ires2:02} {np.sort(dist.flat)[:5]}')
         dot = np.sum(
            rotframes1[:, :, 0].reshape(-1, 1, 4) * rotframes2[:, :, 0].reshape(1, -1, 4), axis=2)
         ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
         ang_delta = np.abs(ang - 109.4712206)

         arg.timer.checkpoint('rotcloud ang')

         err = np.sqrt((ang_delta / angle_to_cart_err_ratio)**2 + dist**2)
         rot1err2 = np.min(err, axis=1)
         bestrot2 = np.argmin(err, axis=1)
         disterr = dist[np.arange(len(bestrot2)), bestrot2]
         angerr = ang_delta[np.arange(len(bestrot2)), bestrot2]
         ok = (rot1err2 < err_tolerance)
         ok *= (angerr < angle_err_tolerance)
         ok *= (disterr < dist_err_tolerance)
         # ok = rot1err2 < err_tolerance
         # ok = disterr < dist_err_tolerance
         # ok = angerr < angle_err_tolerance
         hits1 = np.argwhere(ok).reshape(-1)

         arg.timer.checkpoint('rotcloud match check')

         # hits = (ang_delta < angle_err_tolerance) * (dist < dist_err_tolerance)
         if len(hits1):
            # print(rotcloud1.amino_acid, rotcloud2.amino_acid, ires1, ires2)
            hits2 = bestrot2[hits1]
            hits = np.stack([hits1, hits2], axis=1)
            # print(
            # f'stats {ires1:2} {ires2:2} {np.sum(ang_delta < 10):7} {np.sum(dist < 1.0):5} {np.sum(hits):3}'
            # )
            for ihit, hit in enumerate(hits):
               frame1 = rotframes1[hit[0]]
               frame2 = rotframes2[hit[1]]

               arg.timer.checkpoint('xtal_search')

               parl = (frame1[:, 0] + frame2[:, 0]) / 2.0
               perp = rp.homog.hcross(frame1[:, 0], frame2[:, 0])
               metalaxis1 = rp.homog.hrot(parl, +45) @ perp
               metalaxis2 = rp.homog.hrot(parl, -45) @ perp
               symang1 = rp.homog.line_angle(metalaxis1, spec.pept_axis)
               symang2 = rp.homog.line_angle(metalaxis2, spec.pept_axis)
               match1 = np.abs(np.degrees(symang1) - xspec.dihedral) < sym_axes_angle_tolerance
               match2 = np.abs(np.degrees(symang2) - xspec.dihedral) < sym_axes_angle_tolerance
               if not (match1 or match2): continue
               matchsymang = symang1 if match1 else symang2
               metal_axis = metalaxis1 if match1 else metalaxis2
               if rp.homog.angle(metal_axis, spec.pept_axis) > np.pi / 2:
                  metal_axis[:3] = -metal_axis[:3]
               metal_pos = (rotframes1[hit[0], :, 3] + rotframes2[hit[1], :, 3]) / 2.0

               correction_axis = rp.homog.hcross(metal_axis, spec.pept_axis)
               correction_angle = np.abs(matchsymang - np.radians(xspec.dihedral))
               # print('target', xspec.dihedral, '---------------------')
               # print(np.degrees(correction_angle))
               # print(np.degrees(matchsymang))
               # print('before', rp.homog.angle_degrees(metal_axis, spec.pept_axis))

               for why_do_i_need_this in (-correction_angle, correction_angle):
                  metal_axis_try = rp.homog.hrot(correction_axis, why_do_i_need_this) @ metal_axis
                  if np.allclose(rp.homog.angle_degrees(metal_axis_try, spec.pept_axis),
                                 xspec.dihedral, atol=0.001):
                     metal_axis = metal_axis_try
                     break
               # print('after ', rp.homog.angle_degrees(metal_axis, spec.pept_axis))

               assert np.allclose(rp.homog.angle_degrees(metal_axis, spec.pept_axis),
                                  xspec.dihedral, atol=0.001)

               arg.timer.checkpoint('axes geom checks')

               # pose.dump_pdb('before.pdb')
               pose2mut = mof.util.mutate_two_res(
                  pose, ires1, rotcloud1.amino_acid, rotcloud1.rotchi[hit[0]], ires2,
                  rotcloud2.amino_acid, rotcloud2.rotchi[hit[1]], sym_num)
               # pose2mut.dump_pdb('after.pdb')

               search_spec.sfxn_filter(pose2mut)
               sc_2res = (pose2mut.energies().residue_total_energy(ires1) +
                          pose2mut.energies().residue_total_energy(ires2))
               sc_2res_orig = (pose.energies().residue_total_energy(ires1) +
                               pose.energies().residue_total_energy(ires2))

               arg.timer.checkpoint('mut_two_res')

               # print('scores', sc_2res, sc_2res_orig)

               # if sc_2res > arg.max_2res_score: continue
               if sc_2res - sc_2res_orig > arg.max_2res_score: continue

               tag = ('hit_%s_%s_%i_%i_%i' %
                      (rotcloud1.amino_acid, rotcloud2.amino_acid, ires1, ires2, ihit))

               arg.timer.checkpoint('xtal_search')

               xtal_poses = mof.xtal_build.xtal_build(
                  pdb_name,
                  xspec,
                  pose2mut,
                  peptide_sym,
                  spec.pept_orig,
                  spec.pept_axis,
                  'C2',
                  metal_pos,
                  metal_axis,
                  rpxbody,
                  tag,
                  **arg,
               )
               if not xtal_poses: continue

               # for Xalign, xtal_pose, rpxbody_pdb in xtal_poses:
               # xtal_pose.dump_pdb(tag + '_xtal.pdb')
               # rp.util.dump_str(rpxbody_pdb, tag + '_clashcheck.pdb')
               # assert 0

               arg.timer.checkpoint('xtal_search')

               for ixtal, (xalign, xtal_pose, body_pdb, ncontact,
                           energy) in enumerate(xtal_poses):
                  celldim = xtal_pose.pdb_info().crystinfo().A()
                  fname = f"{pdb_name}_{xspec.spacegroup.replace(' ','_')}_{tag}_cell{int(celldim):03}_ncontact{ncontact:02}_score{int(energy):03}"
                  results.append(
                     mof.result.Result(
                        xspec,
                        fname,
                        xalign,
                        rpxbody,
                        xtal_pose,
                        body_pdb,
                        ncontact,
                        energy,
                     ))

               arg.timer.checkpoint('build_result')

               ### debug crap
               if xtal_poses:
                  print(
                     "HIT",
                     rotcloud1.amino_acid,
                     rotcloud2.amino_acid,
                     ires1,
                     ires2,
                     np.round(np.degrees(matchsymang), 3),
                     hit,
                     rotcloud1.rotchi[hit[0]],
                     rotcloud2.rotchi[hit[1]],
                     # sc_2res,
                     np.round(dist[tuple(hit)], 3),
                     np.round(ang_delta[tuple(hit)], 3),
                     np.round(sc_2res - sc_2res_orig, 3),
                  )
               # rotcloud1.dump_pdb(fn + '_a.pdb', stub1, which=hit[0])
               # rotcloud2.dump_pdb(fn + '_b.pdb', stub2, which=hit[1])
               # rpxbody2.dump_pdb(fn + '_sym.pdb')
               # metalaxispos = metal_pos[:3] + metal_axis[:3] + metal_axis[:3] + metal_axis[:3]
               # pose2mut.dump_pdb(tag + '_before.pdb')
               # hokey_position_atoms(pose2mut, ires1, ires2, metal_pos, metalaxispos)
               # pose2mut.dump_pdb(tag + '_after.pdb')
               # assert 0
               ### end debug crap

   arg.timer.checkpoint('xtal_search')

   return results

def xtal_search_single_residue(search_spec, pose, **arg):
   arg = rp.Bunch(arg)

   spec = search_spec
   xspec = spec.xtal_spec

   results = list()

   p_n = pose.pdb_info().name().split('/')[-1]
   # gets rid of the ".pdb" at the end of the pdb name
   pdb_name = p_n[:-4]

   print(f'{pdb_name} searching')

   # check the symmetry type of the pdb
   last_res = rt.core.pose.chain_end_res(pose).pop()
   total_res = int(last_res)
   sym_num = pose.chain(last_res)
   if sym_num < 2:
      print('bad pdb', p_n)
      return list()
   sym = int(sym_num)
   peptide_sym = "C%i" % sym_num

   for ires in range(1, int(total_res / sym) + 1):
      if pose.residue_type(ires) not in (spec.rts.name_map('GLY'), spec.rts.name_map('ALA'),
                                         spec.rts.name_map('DALA')):
         continue
      lig_poses = util.mut_to_ligand(pose, ires, spec.ligands, spec.sym_of_ligand)
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
            rot_pose = rt.protocols.grafting.return_region(lig_pose, 1, lig_pose.size())

            if arg.debug:
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('1HB'), ires),
                                rVec(0, 0, -2))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('CB'), ires),
                                rVec(0, 0, +0.0))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('2HB'), ires),
                                rVec(0, 0, +2))

            spec.dun_sfxn(rot_pose)
            dun_score = rot_pose.energies().residue_total_energy(ires)
            if dun_score >= spec.max_dun_score:
               bad_rots += 1
               continue
            rpxbody = rp.Body(rot_pose)

            ############ fix this

            metal_orig = hm.hpoint(util.coord_find(rot_pose, ires, 'VZN'))
            hz = hm.hpoint(util.coord_find(rot_pose, ires, 'HZ'))
            ne = hm.hpoint(util.coord_find(rot_pose, ires, 'VNE'))
            metal_his_bond = hm.hnormalized(metal_orig - ne)
            metal_sym_axis0 = hm.hnormalized(hz - metal_orig)
            dihedral = xspec.dihedral

            ############# ...

            rots_around_nezn = hm.xform_around_dof_for_vector_target_angle(
               fix=spec.pept_axis, mov=metal_sym_axis0, dof=metal_his_bond,
               target_angle=np.radians(dihedral))

            for idof, rot_around_nezn in enumerate(rots_around_nezn):
               metal_sym_axis = rot_around_nezn @ metal_sym_axis0
               assert np.allclose(hm.line_angle(metal_sym_axis, spec.pept_axis),
                                  np.radians(dihedral))

               newhz = util.coord_find(rot_pose, ires, 'VZN') + 2 * metal_sym_axis[:3]

               aid = rt.core.id.AtomID(rot_pose.residue(ires).atom_index('HZ'), ires)
               xyz = rVec(newhz[0], newhz[1], newhz[2])
               rot_pose.set_xyz(aid, xyz)

               tag = f'{pdb_name}_{ires}_{lig_poses[lig_pose][0]}_{idof}_{pose_num}'
               xtal_poses = mof.xtal_build.xtal_build(
                  pdb_name,
                  xspec,
                  rot_pose,
                  peptide_sym,
                  spec.pept_orig,
                  spec.pept_axis,
                  lig_sym,
                  metal_orig,
                  metal_sym_axis,
                  rpxbody,
                  tag,
               )

               if False and xtal_poses:
                  print(rot_pose)
                  print(pdb_name)
                  print(xspec)
                  rot_pose.dump_pdb('test_xtal_build_p213.pdb')
                  print(ires)
                  print(peptide_sym)
                  print(spec.pept_orig)
                  print(spec.pept_axis)
                  print(lig_sym)
                  print(metal_orig)
                  print(metal_sym_axis)
                  print('rp.Body(pose)')

                  xalign, xpose, bodypdb = xtal_poses[0]
                  print(xalign)
                  xpose.dump_pdb('xtal_pose.pdb')

                  assert 0

               for ixtal, (xalign, xtal_pose, body_pdb) in enumerate(xtal_poses):
                  celldim = xtal_pose.pdb_info().crystinfo().A()
                  fname = f"{xspec.spacegroup.replace(' ','_')}_cell{int(celldim):03}_{tag}"
                  results.append(
                     mof.result.Result(
                        xspec,
                        fname,
                        xalign,
                        rpxbody,
                        xtal_pose,
                        body_pdb,
                     ))

            pose_num += 1
      # print(pdb_name, 'res', ires, 'bad rots:', bad_rots)
      else:
         continue

   return results

def hokey_position_atoms(pose, ires1, ires2, metal_pos, metalaxispos):
   znres1, znres2 = None, None
   #   if pose.residue(ires1).name() == 'HZD':
   #      znres1 = ires1
   #      znatom1 = 'VZN'
   #      axisatom1 = 'HZ'
   #   elif pose.residue(ires1).name3() == 'CYS':
   #      znres1 = ires1
   #      znatom1 = 'HG'
   #      axisatom1 = '2HB'
   #   elif pose.residue(ires1).name3() == 'ASP':
   #      znres1 = ires1
   #      znatom1 = '1HB'
   #      axisatom1 = '2HB'
   #   elif pose.residue(ires1).name3() == 'GLU':
   #      znres1 = ires1
   #      znatom1 = '1HG'
   #      axisatom1 = '2HG'
   #
   #   if pose.residue(ires2).name3() == 'HZD':
   #      znres2 = ires2
   #      znatom2 = 'VZN'
   #      axisatom2 = 'HZ'
   #   elif pose.residue(ires2).name3() == 'CYS':
   #      znres2 = ires2
   #      znatom2 = 'HG'
   #      axisatom2 = '2HB'
   #   elif pose.residue(ires2).name3() == 'ASP':
   #      znres2 = ires2
   #      znatom2 = '1HB'
   #      axisatom2 = '2HB'
   #   elif pose.residue(ires2).name3() == 'GLU':
   #      znres2 = ires2
   #      znatom2 = '1HG'
   #      axisatom2 = '2HG'
   znres1 = ires1
   znres2 = ires2
   znatom1 = '1HB'
   znatom2 = '1HB'
   axisatom1 = '2HB'
   axisatom2 = '2HB'
   for znres, znatom, axisatom in [(znres1, znatom1, axisatom1), (znres2, znatom2, axisatom2)]:
      pose.set_xyz(AtomID(pose.residue(znres).atom_index(znatom), znres),
                   rVec(metal_pos[0], metal_pos[1], metal_pos[2]))
      pose.set_xyz(AtomID(pose.residue(znres).atom_index(axisatom), znres),
                   rVec(metalaxispos[0], metalaxispos[1], metalaxispos[2]))
