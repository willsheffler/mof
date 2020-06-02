import os, mof, numpy as np, rpxdock as rp
from mof.pyrosetta_init import (rosetta as r, rts, makelattice, addcst_dis, addcst_ang,
                                addcst_dih, name2aid, printscores)
from pyrosetta import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec

def print_nonzero_energies(sfxn, pose):
   for st in sfxn.get_nonzero_weighted_scoretypes():
      print(st)

def minimize_mof_xtal(sfxn, xspec, pose, **kw):
   kw = rp.Bunch(kw)
   try:
      print('''
minimize.py score initial....................      687.825
minimize.py: score before chem bonds.........      687.825
minimize.py: score after chem bonds..........      166.551
minimize.py: score after chainbreaks.........      166.551
minimize.py: score after metal olap .........      166.551
minimize.py: score after metal dist .........      176.766
minimize.py: score after metal dir...........      275.546
minimize.py: score after lig angle added.....      290.724
minimize.py: score after min no scale........      111.465
==========================================================
'''.strip())
      os.remove('before.pdb')
      os.remove('zafter.pdb')
   except:
      pass

   nresasym = pose.size()
   beg = 1
   end = nresasym - 1
   metalres = rts.name_map('ZN')
   metalname = 'ZN'
   metalresnos = [nresasym, 2 * nresasym]  # TODO.. make not stupid
   metalnbonds = 4
   metalligang = np.radians(109.47)
   metalligdist = 2.2
   metalaid = AtomID(1, metalresnos[0])
   sd_metal_olap = 0.01
   sd_metal_dir = 0.5
   sd_metal_lig_dist = 0.3
   sd_metal_lig_ang = 1.6
   sd_metal_coo = 0.8
   sd_cut_dis = 0.01
   sd_cut_ang = 0.01
   sd_cut_dih = 0.1

   sfxn = sfxn.clone()
   # fa_elec is useless whin the ZN is virtual -- missing charge
   sfxn.set_weight(r.core.scoring.ScoreType.fa_elec, 0.0)
   sfxn.set_weight(r.core.scoring.ScoreType.atom_pair_constraint, 1.0)
   sfxn.set_weight(r.core.scoring.ScoreType.angle_constraint, 1.0)
   sfxn.set_weight(r.core.scoring.ScoreType.dihedral_constraint, 1.0)

   pose = pose.clone()
   r.core.pose.remove_lower_terminus_type_from_pose_residue(pose, beg)
   r.core.pose.remove_upper_terminus_type_from_pose_residue(pose, end)
   for ir in range(1, pose.size() + 1):
      if 'HIS' in pose.residue(ir).name():
         newname = pose.residue(ir).name().replace('HIS', 'HIS_D')
         newname = newname.replace('_D_D', '')
         r.core.pose.replace_pose_residue_copying_existing_coordinates(
            pose, ir, rts.name_map(newname))

   if True:
      tmp = pose.clone()
      r.core.pose.replace_pose_residue_copying_existing_coordinates(tmp, metalresnos[0], metalres)
      makelattice(tmp)
      tmp.dump_pdb('before.pdb')
   makelattice(pose)
   print(f'minimize.py score initial.................... {sfxn(pose):10.3f}')
   # print_nonzero_energies(sfxn, pose)

   syminfo = r.core.pose.symmetry.symmetry_info(pose)
   symdofs = syminfo.get_dofs()
   allowed_jumps = list()
   # for jid, dofinfo in symdofs.items():
   #    # if dofinfo.allow_dof(1) and not any(dofinfo.allow_dof(i) for i in range(2, 7)):
   #    allowed_jumps.append(jid)
   # print('minimize_mof_xtal allowed jumps:', allowed_jumps)
   nxyz = pose.residue(beg).xyz('N')
   cxyz = pose.residue(end).xyz('C')
   nac, cac = None, None  # (N/C)-(a)djacent (c)hain
   for isub in range(1, syminfo.subunits()):
      othern = pose.residue((isub + 0) * nresasym + 1).xyz('N')
      otherc = pose.residue((isub + 1) * nresasym - 1).xyz('C')
      if nxyz.distance(otherc) < 2.0: cac = (isub + 1) * nresasym - 1
      if cxyz.distance(othern) < 2.0: nac = (isub + 0) * nresasym + 1
   assert nac and cac, 'backbone is weird?'
   print('peptide connection 1:', cac, beg)
   print('peptide_connection 2:', end, nac)
   # pose.dump_pdb('check_cuts.pdb')
   # assert 0

   f_metal_olap = r.core.scoring.func.HarmonicFunc(0.0, sd_metal_olap)
   f_point_at_metal = r.core.scoring.func.HarmonicFunc(0.0, sd_metal_dir)
   f_metal_coo = r.core.scoring.func.CircularHarmonicFunc(0.0, sd_metal_coo)
   f_metal_lig_dist = r.core.scoring.func.HarmonicFunc(metalligdist, sd_metal_lig_dist)
   f_metal_lig_ang = r.core.scoring.func.HarmonicFunc(metalligang, sd_metal_lig_ang)
   f_cut_dis = r.core.scoring.func.HarmonicFunc(1.328685, sd_cut_dis)
   f_cut_ang_cacn = r.core.scoring.func.HarmonicFunc(2.028, sd_cut_ang)
   f_cut_ang_cnca = r.core.scoring.func.HarmonicFunc(2.124, sd_cut_ang)
   f_cut_dih = r.core.scoring.func.CircularHarmonicFunc(np.pi, sd_cut_dih)
   f_cut_dihO = r.core.scoring.func.CircularHarmonicFunc(0.00, sd_cut_dih)

   ################### check cutpoint ##################

   conf = pose.conformation().clone()
   assert r.core.conformation.symmetry.is_symmetric(conf)
   # print(pose.pdb_info().crystinfo())
   pi = pose.pdb_info()
   # conf.detect_bonds()
   conf.declare_chemical_bond(cac, 'C', beg, 'N')
   # conf.declare_chemical_bond(end, 'N', nac, 'N')
   pose.set_new_conformation(conf)
   pose.set_new_energies_object(r.core.scoring.symmetry.SymmetricEnergies())
   pose.pdb_info(pi)
   print(f'minimize.py: score after chem bonds.......... {sfxn(pose):10.3f}')

   ############### chainbreaks ################3

   # this doesn't behave well...
   # # 39 C / 1 OVU1
   # # 39 OVL1 / 1 N
   # # 29 OVL2 / CA
   # r.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_UPPER', beg)
   # r.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_LOWER', end)
   # cres1 = pose.residue(cac)
   # nres1 = pose.residue(1)
   # cres2 = pose.residue(end)
   # nres2 = pose.residue(nac)
   # # Apc = r.core.scoring.constraints.AtomPairConstraint
   # # getaid = lambda p, n, i: AtomID(p.residue(i).atom_index(n.strip()), i)
   # for cst in [
   #       Apc(getaid(pose, 'C   ', cac), getaid(pose, 'OVU1', beg), f_cut),
   #       Apc(getaid(pose, 'OVL1', cac), getaid(pose, 'N   ', beg), f_cut),
   #       Apc(getaid(pose, 'OVL2', cac), getaid(pose, 'CA  ', beg), f_cut),
   #       Apc(getaid(pose, 'C   ', end), getaid(pose, 'OVU1', nac), f_cut),
   #       Apc(getaid(pose, 'OVL1', end), getaid(pose, 'N   ', nac), f_cut),
   #       Apc(getaid(pose, 'OVL2', end), getaid(pose, 'CA  ', nac), f_cut),
   # ]:
   #    pose.add_constraint(cst)

   addcst_dis(pose, cac, 'C ', beg, 'N', f_cut_dis)
   addcst_dis(pose, end, 'C ', nac, 'N', f_cut_dis)
   print(f'minimize.py: score after chainbreak dis...... {sfxn(pose):10.3f}')
   addcst_ang(pose, cac, 'CA', cac, 'C', beg, 'N ', f_cut_ang_cacn)
   addcst_ang(pose, cac, 'C ', beg, 'N', beg, 'CA', f_cut_ang_cnca)
   addcst_ang(pose, end, 'CA', end, 'C', nac, 'N ', f_cut_ang_cacn)
   addcst_ang(pose, end, 'C ', nac, 'N', nac, 'CA', f_cut_ang_cnca)
   print(f'minimize.py: score after chainbreak ang...... {sfxn(pose):10.3f}')
   # print(r.numeric.dihedral(
   #       pose.residue(cac).xyz('CA'),
   #       pose.residue(cac).xyz('C'),
   #       pose.residue(beg).xyz('N'),
   #       pose.residue(beg).xyz('CA'),
   #    ))
   addcst_dih(pose, cac, 'CA', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dih)
   addcst_dih(pose, end, 'CA', end, 'C', nac, 'N ', nac, 'CA', f_cut_dih)
   addcst_dih(pose, cac, 'O ', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dihO)
   addcst_dih(pose, end, 'O ', end, 'C', nac, 'N ', nac, 'CA', f_cut_dihO)
   print(f'minimize.py: score after chainbreak dihedral. {sfxn(pose):10.3f}')

   ############## metal constraints ################

   # metal ligand distance
   for i, j in [(i, j) for i in metalresnos for j in metalresnos if i < j]:
      addcst_dis(pose, i, metalname, j, metalname, f_metal_olap)
   print(f'minimize.py: score after metal olap ......... {sfxn(pose):10.3f}')

   allowed_elems = 'NOS'
   znpos = pose.residue(metalresnos[0]).xyz(1)
   znbonded = list()
   for ir in range(1, len(pose.residues) + 1):
      res = pose.residue(ir)
      if not res.is_protein(): continue
      for ia in range(5, res.nheavyatoms() + 1):
         aid = AtomID(ia, ir)
         elem = res.atom_name(ia).strip()[0]
         if elem in allowed_elems:
            xyz = pose.xyz(aid)
            dist = xyz.distance(znpos)
            if dist < 3.5:
               znbonded.append(aid)
   if len(znbonded) != metalnbonds:
      print('WRONG NO OF LIGANDING ATOMS', len(znbonded))
      return None

   # metal/lig distance constraints
   for i, aid in enumerate(znbonded):
      pose.add_constraint(
         r.core.scoring.constraints.AtomPairConstraint(metalaid, aid, f_metal_lig_dist))

   print(f'minimize.py: score after metal dist ......... {sfxn(pose):10.3f}')

   # for aid in znbonded:
   #    print(aid.rsd(), aid.atomno(),
   #          pose.residue(aid.rsd()).name(),
   #          pose.residue(aid.rsd()).atom_name(aid.atomno()))
   # assert 0

   # lig/metal/lig angle constraints (or dihedral in-place constraint for COO)
   # TODO kinda hacky... will need to be more general?
   for i, aid in enumerate(znbonded):
      ir, res = aid.rsd(), pose.residue(aid.rsd())
      if all(_ not in res.name() for _ in 'ASP CYS HIS GLU'.split()):
         assert 0, f'unrecognized res {res.name()}'
      if any(_ in res.name() for _ in 'ASP GLU'.split()):
         # metal comes off of OD1/OE1
         ir, coo = aid.rsd(), ('OD1 CG OD2' if 'ASP' in res.name() else 'OE1 CD OE2').split()
         addcst_dih(pose, ir, coo[0], ir, coo[1], ir, coo[2], metalaid.rsd(), metalname,
                    f_metal_coo)
      else:
         if 'HIS' in res.name(): aname = 'HD1' if res.has('HD1') else 'HE2'
         if 'CYS' in res.name(): anmae = 'HG'
         addcst_ang(pose, ir, res.atom_name(aid.atomno()), metalaid.rsd(), metalname, ir, aname,
                    f_point_at_metal)
         # cst = r.core.scoring.constraints.AngleConstraint(aid, metalaid, aid2, f_point_at_metal)
         # pose.add_constraint(cst)
   print(f'minimize.py: score after metal dir........... {sfxn(pose):10.3f}')

   for i, iaid in enumerate(znbonded):
      for j, jaid in enumerate(znbonded[:i]):
         # pripnt(i, j)
         cst = r.core.scoring.constraints.AngleConstraint(iaid, metalaid, jaid, f_metal_lig_ang)
         pose.add_constraint(cst)

   print(f'minimize.py: score after lig angle added..... {sfxn(pose):10.3f}')

   ################ minimization #########################

   movemap = r.core.kinematics.MoveMap()
   movemap.set_bb(True)
   movemap.set_chi(True)
   movemap.set_jump(False)
   for i in allowed_jumps:
      movemap.set_jump(True, i)
   mover = r.protocols.minimization_packing.symmetry.SymMinMover(
      movemap, sfxn, 'lbfgs_armijo_nonmonotone', 0.01, True)  # tol, nblist

   mover.apply(pose)
   print(f'minimize.py: score after min no scale........ {sfxn(pose):10.3f}')

   printscores(sfxn, pose)

   kw.timer.checkpoint(f'min scale 1.0')

   asym = r.core.pose.Pose()
   r.core.pose.symmetry.extract_asymmetric_unit(pose, asym, False)
   r.core.pose.replace_pose_residue_copying_existing_coordinates(asym, metalresnos[0], metalres)

   for ir in metalresnos:
      r.core.pose.replace_pose_residue_copying_existing_coordinates(pose, ir, metalres)

   pose.dump_pdb('zafter.pdb')

   print(kw.timer)

   assert 0

   return asym
