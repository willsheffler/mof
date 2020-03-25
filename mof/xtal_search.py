import rpxdock as rp, numpy as np
from rpxdock import homog as hm
from mof import xtal_spec
from mof import data, xtal_spec, util
import mof.pyrosetta_init
from mof.xtal_build import xtal_build
from mof.result import Result
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
      self.xtal_spec = xtal_spec.get_xtal_spec(self.spacegroup)
      self.chm = rt.core.chemical.ChemicalManager.get_instance()
      self.rts = self.chm.residue_type_set('fa_standard')
      self.scfxn = rt.core.scoring.ScoreFunction()
      self.scfxn.set_weight(rt.core.scoring.ScoreType.fa_dun, 1.0)

def xtal_search_single_residue(search_spec, pose, debug):

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

            if debug:
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('1HB'), ires),
                                xyzVec(0, 0, -2))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('CB'), ires),
                                xyzVec(0, 0, +0.0))
               rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('2HB'), ires),
                                xyzVec(0, 0, +2))

            spec.scfxn(rot_pose)
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
               xtal_poses = xtal_build(pdb_name, xspec, rot_pose, ires, peptide_sym,
                                       spec.pept_orig, spec.pept_axis, lig_sym, metal_origin,
                                       metal_sym_axis, rpxbody, tag)

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
                  results.append(Result(xspec, fname, xalign, rpxbody, xtal_pose, body_pdb))

            pose_num += 1
      # print(pdb_name, 'res', ires, 'bad rots:', bad_rots)
      else:
         continue

   return results

def xtal_search_two_residues(search_spec, pose, rotcloud1, rotcloud2, debug):
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

   rpxbody = rp.Body(pose)

   target_dot = np.cos(np.radians(107.0))

   for ires in range(1, int(total_res) + 1):
      # if pose.residue_type(ires) not in (spec.rts.name_map('ALA'), spec.rts.name_map('DALA')):
      if pose.residue_type(ires) != spec.rts.name_map('ALA'):
         continue
      stub1 = rpxbody.stub[ires - 1]
      rotframes1 = stub1 @ rotcloud1.rotframes
      # rotcloud1.dump_pdb(f'cloud_a_{ires:02}.pdb', position=stub1)
      for jres in range(1, int(total_res / sym) + 1):
         if pose.residue_type(jres) != spec.rts.name_map('ALA'):
            continue
         stub2 = rpxbody.stub[jres - 1]
         # rotcloud2.dump_pdb(f'cloud_b_{jres:02}.pdb', position=stub2)
         if ires == jres: continue
         rotframes2 = stub2 @ rotcloud2.rotframes

         dist = rotframes1[:, :, 3].reshape(-1, 1, 4) - rotframes2[:, :, 3].reshape(1, -1, 4)
         dist = np.linalg.norm(dist, axis=2)
         # if np.min(dist) < 1.0:
         # print(f'{ires:02} {jres:02} {np.sort(dist.flat)[:5]}')
         dot = np.sum(
            rotframes1[:, :, 0].reshape(-1, 1, 4) * rotframes2[:, :, 0].reshape(1, -1, 4), axis=2)
         ang = np.degrees(np.arccos(dot))
         angerr = np.abs(ang - 107)
         hits = (angerr < 15.0) * (dist < 1.0)
         if np.sum(hits):
            print(
               f'{ires:2} {jres:2} {np.sum(angerr < 10):7} {np.sum(dist < 1.0):5} {np.sum(hits):3}'
            )
            hits = np.argwhere(hits)
            print(dist.shape, len(rotcloud1), len(rotcloud2))
            print(hits[0, 0], hits[0, 1])
            rotcloud1.dump_pdb('test1.pdb', stub1, which=hits[0, 0])
            rotcloud2.dump_pdb('test2.pdb', stub2, which=hits[0, 1])
            rpxbody.dump_pdb('test.pdb')
            assert 0

   assert 0
   # rpxbody.dump_pdb('rpxbody.pdb')

   return results
