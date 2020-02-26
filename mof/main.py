import sys, numpy as np, rpxdock as rp, rpxdock.homog as hm

from mof import data, xtal_spec, util
from mof.xtal_build import xtal_build
from mof.result import Result
import mof.pyrosetta_init

from pyrosetta import rosetta as rt, init as pyrosetta_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec

_DEBUG = False

def main():

   max_dun_score = 4.0

   sym_of_ligand = dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2')

   # ligands = ['HZ3', 'DHZ3']
   # xspec = xtal_spec.get_xtal_spec('p213')

   ligands = ['HZ4', 'DHZ4']
   xspec = xtal_spec.get_xtal_spec('f432')

   # ligands = ['HZD', 'DHZD']
   # xspec = xtal_spec.get_xtal_spec(None)

   pept_orig = np.array([0, 0, 0, 1])
   pept_axis = np.array([0, 0, 1, 0])

   if len(sys.argv) > 1:
      pdb_gen = util.gen_pdbs(sys.argv[1])
   else:
      # test
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!! no pdb list input, using test "only_one" !!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      pdb_gen = util.gen_pdbs('mof/tests/only_one/input/only_one.list')
   prepped_pdb_gen = util.prep_poses(pdb_gen)

   chm = rt.core.chemical.ChemicalManager.get_instance()
   rts = chm.residue_type_set('fa_standard')
   scfxn = rt.core.scoring.ScoreFunction()
   scfxn.set_weight(rt.core.scoring.ScoreType.fa_dun, 1.0)

   results = list()

   for pdb in prepped_pdb_gen:
      # gets the pdb name for outputs later
      p_n = pdb.pdb_info().name().split('/')[-1]
      # gets rid of the ".pdb" at the end of the pdb name
      pdb_name = p_n[:-4]

      print(f'{pdb_name} searching')

      # check the symmetry type of the pdb
      last_res = rt.core.pose.chain_end_res(pdb).pop()
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
                  rot_pose = rt.protocols.grafting.return_region(lig_pose, 1, lig_pose.size())

                  if _DEBUG:
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('1HB'), ires),
                                      xyzVec(0, 0, -2))
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('CB'), ires),
                                      xyzVec(0, 0, +0.0))
                     rot_pose.set_xyz(AtomID(rot_pose.residue(ires).atom_index('2HB'), ires),
                                      xyzVec(0, 0, +2))

                  scfxn(rot_pose)
                  dun_score = rot_pose.energies().residue_total_energies(ires)[
                     rt.core.scoring.ScoreType.fa_dun]
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

                     aid = rt.core.id.AtomID(rot_pose.residue(ires).atom_index('HZ'), ires)
                     xyz = xyzVec(newhz[0], newhz[1], newhz[2])
                     rot_pose.set_xyz(aid, xyz)

                     tag = f'{pdb_name}_{ires}_{lig_poses[lig_pose][0]}_{idof}_{pose_num}'
                     xtal_poses = xtal_build(pdb_name, xspec, rot_pose, ires, peptide_sym,
                                             pept_orig, pept_axis, lig_sym, metal_origin,
                                             metal_sym_axis, rpxbody, tag)

                     if xtal_poses:
                        print(rot_pose)
                        print(pdb_name)
                        print(xspec)
                        rot_pose.dump_pdb('test_xtal_build_p213.pdb')
                        print(ires)
                        print(peptide_sym)
                        print(pept_orig)
                        print(pept_axis)
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

   if not results:
      return

   xforms = np.array([r.xalign for r in results])
   non_redundant = rp.filter.filter_redundancy(xforms, results[0].rpxbody, every_nth=1,
                                               max_bb_redundancy=1.0, max_cluster=10000)
   for i, result in enumerate(results):
      if i in non_redundant:
         print('dumping', result.label)
         result.xtal_asym_pose.dump_pdb('asym_' + result.label + '.pdb')
         rp.util.dump_str(result.symbody_pdb, 'sym_' + result.label + '.pdb')

   print("DONE")

if __name__ == '__main__':
   main()
