import mof, rpxdock as rp, numpy as np
from rpxdock import homog as hm

import mof.pyrosetta_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec

from pyrosetta import rosetta as rt, init as pyrosetta_init

class XtalSearchSpec(object):
   """stuff needed for pepdide xtal search"""
   def __init__(self, spacegroup, pept_axis, pept_orig, ligands, sym_of_ligand, max_dun_score):
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
      self.rep_sfxn = rt.core.scoring.ScoreFunction()
      self.rep_sfxn.set_weight(rt.core.scoring.ScoreType.fa_rep, 1.0)

def xtal_search_two_residues(search_spec, pose, rotcloud1base, rotcloud2base, err_tolerance=1.5,
                             min_dist_to_z_axis=6.0, sym_axes_angle_tolerance=3.0, **arg):
   arg = rp.Bunch(arg)

   spec = search_spec
   xspec = spec.xtal_spec

   results = list()

   farep_orig = search_spec.rep_sfxn(pose)

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

   rpxbody = rp.Body(pose)

   for ires1 in range(1, int(total_res / sym) + 1):
      # if pose.residue_type(ires1) not in (spec.rts.name_map('ALA'), spec.rts.name_map('DALA')):
      if pose.residue_type(ires1) != spec.rts.name_map('ALA'):
         continue
      stub1 = rpxbody.stub[ires1 - 1]

      rots1ok = min_dist_to_z_axis < np.linalg.norm(
         (stub1 @ rotcloud1base.rotframes)[:, :2, 3], axis=1)
      if 0 == np.sum(rots1ok): continue
      rotcloud1 = rotcloud1base.subset(rots1ok)
      rotframes1 = stub1 @ rotcloud1.rotframes
      # rotcloud1.dump_pdb(f'cloud_a_{ires1:02}.pdb', position=stub1)

      range2 = range(1, int(total_res) + 1)
      if rotcloud1base is rotcloud2base: range2 = range(ires1 + 1, int(total_res) + 1)
      for ires2 in range2:
         if ires1 == ires2: continue
         if pose.residue_type(ires2) != spec.rts.name_map('ALA'):
            continue
         stub2 = rpxbody.stub[ires2 - 1]
         # rotcloud2.dump_pdb(f'cloud_b_{ires2:02}.pdb', position=stub2)

         rots2ok = min_dist_to_z_axis < np.linalg.norm(
            (stub2 @ rotcloud2base.rotframes)[:, :2, 3], axis=1)
         if 0 == np.sum(rots2ok): continue
         rotcloud2 = rotcloud2base.subset(rots2ok)
         rotframes2 = stub2 @ rotcloud2.rotframes

         dist = rotframes1[:, :, 3].reshape(-1, 1, 4) - rotframes2[:, :, 3].reshape(1, -1, 4)
         dist = np.linalg.norm(dist, axis=2)
         # if np.min(dist) < 1.0:
         # print(f'{ires1:02} {ires2:02} {np.sort(dist.flat)[:5]}')
         dot = np.sum(
            rotframes1[:, :, 0].reshape(-1, 1, 4) * rotframes2[:, :, 0].reshape(1, -1, 4), axis=2)
         ang = np.degrees(np.arccos(dot))
         angerr = np.abs(ang - 107)
         err = angerr / 15.0 + dist
         rot1err = np.min(err, axis=1)
         bestrot2 = np.argmin(err, axis=1)
         hits1 = np.argwhere(rot1err < err_tolerance).reshape(-1)

         # hits = (angerr < angle_err_tolerance) * (dist < dist_err_tolerance)
         if len(hits1):
            # print(rotcloud1.amino_acid, rotcloud2.amino_acid, ires1, ires2)
            hits2 = bestrot2[hits1]
            hits = np.stack([hits1, hits2], axis=1)
            # print(
            # f'stats {ires1:2} {ires2:2} {np.sum(angerr < 10):7} {np.sum(dist < 1.0):5} {np.sum(hits):3}'
            # )
            for ihit, hit in enumerate(hits):
               frame1 = rotframes1[hit[0]]
               frame2 = rotframes2[hit[1]]

               parl = (frame1[:, 0] + frame2[:, 0]) / 2.0
               perp = rp.homog.hcross(frame1[:, 0], frame2[:, 0])
               metalaxis1 = rp.homog.hrot(parl, +45) @ perp
               metalaxis2 = rp.homog.hrot(parl, -45) @ perp
               symang1 = rp.homog.line_angle(metalaxis1, spec.pept_axis)
               symang2 = rp.homog.line_angle(metalaxis2, spec.pept_axis)
               match11 = np.abs(np.degrees(symang1) - 35.26) < sym_axes_angle_tolerance
               match12 = np.abs(np.degrees(symang2) - 35.26) < sym_axes_angle_tolerance
               match21 = np.abs(np.degrees(symang1) - 54.735) < sym_axes_angle_tolerance
               match22 = np.abs(np.degrees(symang2) - 54.735) < sym_axes_angle_tolerance
               if not (match11 or match12 or match12 or match22): continue
               matchsymang = symang1 if (match11 or match21) else symang2

               # metal_pos = (rotframes1[hit[0], :3, 3] + rotframes2[hit[1], :3, 3]) / 2.0
               # metal_dist_z = np.sqrt(metal_pos[0]**2 + metal_pos[1]**2)
               # if metal_dist_z < 4.0: continue
               pose2 = mof.util.mutate_two_res(
                  pose,
                  ires1,
                  rotcloud1.amino_acid,
                  rotcloud1.rotchi[hit[0]],
                  ires2,
                  rotcloud2.amino_acid,
                  rotcloud2.rotchi[hit[1]],
               )
               farep_delta = search_spec.rep_sfxn(pose2) - farep_orig
               if farep_delta > 1.0: continue
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
                  # farep_delta,
                  np.round(dist[tuple(hit)], 3),
                  np.round(angerr[tuple(hit)], 3),
               )
               fn = ('hit_%s_%s_%i_%i_%i.pdb' %
                     (rotcloud1.amino_acid, rotcloud2.amino_acid, ires1, ires2, ihit))
               rotcloud1.dump_pdb(fn + '_a.pdb', stub1, which=hit[0])
               rotcloud2.dump_pdb(fn + '_b.pdb', stub2, which=hit[1])
               pose2.dump_pdb(fn)

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
      if pose.residue_type(ires) not in (spec.rts.name_map('ALA'), spec.rts.name_map('DALA')):
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
                                xyzVec(0, 0, -2))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('CB'), ires),
                                xyzVec(0, 0, +0.0))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('2HB'), ires),
                                xyzVec(0, 0, +2))

            spec.dun_sfxn(rot_pose)
            dun_score = rot_pose.energies().residue_total_energies(ires)[
               rt.core.scoring.ScoreType.fa_dun]
            if dun_score >= spec.max_dun_score:
               bad_rots += 1
               continue
            rpxbody = rp.Body(rot_pose)

            ############ fix this

            metal_origin = hm.hpoint(util.coord_find(rot_pose, ires, 'VZN'))
            hz = hm.hpoint(util.coord_find(rot_pose, ires, 'HZ'))
            ne = hm.hpoint(util.coord_find(rot_pose, ires, 'VNE'))
            metal_his_bond = hm.hnormalized(metal_origin - ne)
            metal_sym_axis0 = hm.hnormalized(hz - metal_origin)
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
               xyz = xyzVec(newhz[0], newhz[1], newhz[2])
               rot_pose.set_xyz(aid, xyz)

               tag = f'{pdb_name}_{ires}_{lig_poses[lig_pose][0]}_{idof}_{pose_num}'
               xtal_poses = mof.xtal_build.xtal_build(
                  pdb_name,
                  xspec,
                  rot_pose,
                  ires,
                  peptide_sym,
                  spec.pept_orig,
                  spec.pept_axis,
                  lig_sym,
                  metal_origin,
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
                  print(metal_origin)
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
