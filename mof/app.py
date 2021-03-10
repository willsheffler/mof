import mof

import rpxdock as rp
import numpy as np
import sys, os, shutil

from mof.pyrosetta_init import (rosetta, rts, makelattice, addcst_dis, addcst_ang, addcst_dih,
                                name2aid, printscores, get_sfxn, xform_pose, make_residue)
from mof.util import is_rosetta_stuff, strip_rosetta_content_from_results, align_cx_pose_to_z
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta import AtomID
import pyrosetta

def options_setup(get_test_kw=None, verbose=True):
   kw = mof.options.get_cli_args()
   if kw.test_run:
      if get_test_kw: kw = get_test_kw(kw)
      else: raise InputError(f'--test_run but no get_test_kw specified')

   if verbose:
      print_options(kw)

   kw.timer = rp.Timer().start()

   if kw.overwrite and os.path.exists(os.path.dirname(kw.output_prefix)):
      print('overwrite', kw.output_prefix)
      shutil.rmtree(os.path.dirname(kw.output_prefix))
   assert kw.output_prefix

   outdir = os.path.dirname(kw.output_prefix)
   if not outdir:
      outdir = kw.output_prefix + '/'
      # kw.output_prefix = kw.output_prefix + '\n' # this was a nice one!
      kw.output_prefix = kw.output_prefix + '/' + kw.output_prefix.lstrip('_')
   basename = os.path.basename(kw.output_prefix)
   if not basename.strip():
      kw.output_prefix += 'mof'

   print(f'os.makedirs({outdir + "clustered/"}, exist_ok=True)')
   os.makedirs(outdir + 'clustered/', exist_ok=True)

   if not kw.spacegroups:
      print('NO spacegroups specified (e.g. --spacegroups P23)')
      return
   # for k, v in kw.items():
   #    try:
   #       print(k, v)
   #    except ValueError:
   #       print(k, type(v))
   # kw.aa_labels = 'ASP DASP CYS DCYS HIS DHIS HISD DHISD GLU DGLU'.split()
   # kw.aa_labels = 'ASP DASP CYS DCYS HIS DHIS GLU DGLU'.split()
   if 'HISD' in kw.aa_labels:
      print('removing HISD, crashes for some reason')
      kw.aa_labels.remove('HISD')
   if 'DHISD' in kw.aa_labels:
      print('removing DHISD, crashes for some reason')
      kw.aa_labels.remove('DHISD')

   if kw.strip_rosetta_content_from_results: return strip_rosetta_content_from_results(kw)

   return kw

def print_options(kw):

   print()
   print('!' * 100)
   print()
   print('COMMAND LINE (maybe with whitespace changes):')
   print()
   print(' '.join(sys.argv))
   print()
   print('!' * 100)
   print()

   kw = kw.sub(inputs=None, timer=None)
   longest_key = max(len(k) for k in kw) + 2
   longest_val = max([
      len(repr(v) if isinstance(v, (int, float, str, list)) else str(type(v)))
      for v in kw.values()
   ])
   longest_line = 0
   reprs = list()
   for k, v in kw.items():
      vstr = repr(v) if isinstance(v, (int, float, str, list)) else str(type(v))
      sep1 = '.' * (longest_key - len(k))
      sep2 = ' ' * (longest_val - len(vstr))
      reprs.append((k, sep1, vstr, sep2))
      linelen = len(str(k + sep1 + vstr))
      longest_line = max(longest_line, linelen)

   msg1 = '  THESE OPTION REJECT YOUR REALITY  '
   msg2 = '  AND SUBSTITUTE THEIR OWN  '
   sepline = '  !!         ' + ' ' * (3 + longest_line) + '  !!'
   sepline = sepline + os.linesep + sepline
   print()
   print(f"  {'':!^{longest_line + 18}}")
   print(f"  {msg1:!^{longest_line + 18}}")
   print(f"  {msg2:!^{longest_line + 18}}")
   print(f"  {'':!^{longest_line + 18}}")
   print(sepline)
   for k, sep1, vstr, sep2 in reprs:
      print('  !!       ', k, sep1, vstr, sep2, '  !!')
   print(sepline)
   print(f"  {' END OF OPTIONS ':!^{longest_line + 18}}")
   print()

def print_nonzero_energies(sfxn, pose):
   for st in sfxn.get_nonzero_weighted_scoretypes():
      print(st, pose.energies().total_energies()[st])

_bestscores = dict()

def record_nonzero_energies(sfxn, pose):
   for st in sfxn.get_nonzero_weighted_scoretypes():
      if st not in _bestscores:
         _bestscores[st] = pose.energies().total_energies()[st]
      _bestscores[st] = min(_bestscores[st], pose.energies().total_energies()[st])

def set_cell_params(xtal_pose, xspec, cell_spacing):
   ci = pyrosetta.rosetta.core.io.CrystInfo()
   ci.A(abs(cell_spacing))  # cell dimensions
   ci.B(abs(cell_spacing))
   ci.C(abs(cell_spacing))
   ci.alpha(90)  # cell angles
   ci.beta(90)
   ci.gamma(90)
   ci.spacegroup(xspec.spacegroup)  # sace group
   pi = pyrosetta.rosetta.core.pose.PDBInfo(xtal_pose)
   pi.set_crystinfo(ci)
   xtal_pose.pdb_info(pi)

lblmap = dict(
   CYS='lC',
   DCYS='dC',
   ASP='lD',
   DASP='dD',
   GLU='lE',
   DGLU='dE',
   HIS='lH',
   DHIS='dH',
   HISD='lJ',
   DHISD='dJ',
   BPY='lB',
)

def prepare_result(
   xtal_pose_min,
   xspec,
   tag,
   xalign,
   pdb_name,
   mininfo,
   solv_frac,
   iaa,
   ncontact=0,
   enonbonded=0,
   **kw,
):
   kw = rp.Bunch(kw)
   celldim = xtal_pose_min.pdb_info().crystinfo().A()
   label = f"{pdb_name}_{xspec.spacegroup.replace(' ','_')}_{tag}_cell{int(celldim):03}_ncontact{ncontact:02}_score{int(enonbonded):03}"

   info = mininfo.sub(  # adding to mininfo
      label=label,
      xalign=xalign,
      # ncontact=ncontact,
      # enonbonded=enonbonded,
      sequence=','.join(r.name() for r in xtal_pose_min.residues),
      solv_frac=solv_frac,
      celldim=celldim,
      spacegroup=xspec.spacegroup,
      nsubunits=xspec.nsubs,
      nres=xtal_pose_min.size() - 1,
      tag=tag,
   )
   bbcoords = np.array([(v[0], v[1], v[2]) for v in [[r.xyz(n)
                                                      for n in ('N', 'CA', 'C')]
                                                     for r in xtal_pose_min.residues[:-1]]])
   # allow for different num of res
   bbpad = 9e9 * np.ones(shape=(kw.max_pept_size - xtal_pose_min.size() + 1, 3, 3))
   # separate aas in hacky way
   bbpad[-1] = 10000 * iaa
   info['bbcoords'] = np.concatenate([bbcoords, bbpad])

   return rp.Bunch(
      xspec=xspec,
      # rpxbody=rpxbody,
      # asym_pose=xtal_pose,
      asym_pose_min=xtal_pose_min,
      info=info,
   )

def minimize_oneres(sfxn, xspec, pose, debug=False, **kw):

   debug = False

   kw = rp.Bunch(kw)

   nresasym = pose.size()
   beg = 1
   end = nresasym - 1
   metalres = rts.name_map('ZN')
   metalname = 'ZN'
   metalresno = nresasym
   metalnbonds = 4

   metalaid = AtomID(1, metalresno)

   # cst_ang_metal = 109.47
   # cst_dis_metal = 2.2
   # cst_sd_metal_olap = 0.01
   # cst_sd_metal_dir = 0.4
   # cst_sd_metal_lig_dist = 0.2
   # cst_sd_metal_lig_ang = 0.4
   # cst_sd_metal_coo = 0.5
   # cst_sd_cut_dis = 0.01
   # cst_sd_cut_ang = 0.01
   # cst_sd_cut_dih = 0.1

   pose = pose.clone()
   rosetta.core.pose.remove_lower_terminus_type_from_pose_residue(pose, beg)
   rosetta.core.pose.remove_upper_terminus_type_from_pose_residue(pose, end)
   # WTF is this about?
   # for ir in range(1, pose.size() + 1):
   #    if 'HIS' in pose.residue(ir).name():
   #       newname = pose.residue(ir).name().replace('HIS', 'HIS_D')
   #       newname = newname.replace('_D_D', '')
   #       rosetta.core.pose.replace_pose_residue_copying_existing_coordinates(
   #          pose, ir, rts.name_map(newname))

   if False:
      tmp = pose.clone()
      rosetta.core.pose.replace_pose_residue_copying_existing_coordinates(
         tmp, metalresno, metalres)
      makelattice(tmp)
      tmp.dump_pdb('before.pdb')
   makelattice(pose)
   if debug: print(f'minimize.py score initial.................... {sfxn(pose):10.3f}')
   # mof.app.print_nonzero_energies(sfxn, pose)

   syminfo = rosetta.core.pose.symmetry.symmetry_info(pose)
   syminfo.get_dofs()
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
      if nxyz.distance(otherc) < kw.bb_break_dist: cac = (isub + 1) * nresasym - 1
      if cxyz.distance(othern) < kw.bb_break_dist: nac = (isub + 0) * nresasym + 1
   if not (nac and cac):
      print('backbone is weird? probably subunits too broken')
      print(syminfo.subunits())
      print('nac   ', nac)
      print('othern', othern)
      print('cac   ', cac)
      print('otherc', otherc)
      # assert 0

      # pose.dump_pdb(f'backbone_is_weird.pdb')
      # assert 0

      return None, None
   else:
      print(f'backbone is not weird')
   if debug: print('peptide connection 1:', cac, beg)
   if debug: print('peptide_connection 2:', end, nac)
   # pose.dump_pdb('check_cuts.pdb')
   # assert 0

   f_metal_lig_dist = rosetta.core.scoring.func.HarmonicFunc(kw.cst_dis_metal,
                                                             kw.cst_sd_metal_lig_dist)
   f_metal_lig_ang = rosetta.core.scoring.func.HarmonicFunc(np.radians(kw.cst_ang_metal),
                                                            kw.cst_sd_metal_lig_ang)
   f_metal_olap = rosetta.core.scoring.func.HarmonicFunc(0.0, kw.cst_sd_metal_olap)
   f_point_at_metal = rosetta.core.scoring.func.HarmonicFunc(0.0, kw.cst_sd_metal_dir)
   f_metal_coo = rosetta.core.scoring.func.CircularHarmonicFunc(0.0, kw.cst_sd_metal_coo)
   f_cut_dis = rosetta.core.scoring.func.HarmonicFunc(1.328685, kw.cst_sd_cut_dis)
   f_cut_ang_cacn = rosetta.core.scoring.func.HarmonicFunc(2.028, kw.cst_sd_cut_ang)
   f_cut_ang_cnca = rosetta.core.scoring.func.HarmonicFunc(2.124, kw.cst_sd_cut_ang)
   f_cut_dih = rosetta.core.scoring.func.CircularHarmonicFunc(np.pi, kw.cst_sd_cut_dih)
   f_cut_dihO = rosetta.core.scoring.func.CircularHarmonicFunc(0.00, kw.cst_sd_cut_dih)

   ################### check cutpoint ##################

   conf = pose.conformation().clone()
   assert rosetta.core.conformation.symmetry.is_symmetric(conf)
   # print(pose.pdb_info().crystinfo())
   pi = pose.pdb_info()
   # conf.detect_bonds()
   conf.declare_chemical_bond(cac, 'C', beg, 'N')
   # for iframe, f
   # conf.declare_chemical_bond(end, 'N', nac, 'N')
   pose.set_new_conformation(conf)
   pose.set_new_energies_object(rosetta.core.scoring.symmetry.SymmetricEnergies())
   pose.pdb_info(pi)
   if debug: print(f'minimize.py: score after chem bonds.......... {sfxn(pose):10.3f}')

   #############################################3

   cst_cut, cst_lig_dis, cst_lig_ang, cst_lig_ori = list(), list(), list(), list()

   ############### chainbreaks ################3

   # this doesn't behave well...
   # # 39 C / 1 OVU1
   # # 39 OVL1 / 1 N
   # # 29 OVL2 / CA
   # rosetta.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_UPPER', beg)
   # rosetta.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_LOWER', end)
   # cres1 = pose.residue(cac)
   # nres1 = pose.residue(1)
   # cres2 = pose.residue(end)
   # nres2 = pose.residue(nac)
   # # Apc = rosetta.core.scoring.constraints.AtomPairConstraint
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

   cst_cut.append(addcst_dis(pose, cac, 'C ', beg, 'N', f_cut_dis))
   cst_cut.append(addcst_dis(pose, end, 'C ', nac, 'N', f_cut_dis))
   if debug: print(f'minimize.py: score after chainbreak dis...... {sfxn(pose):10.3f}')
   cst_cut.append(addcst_ang(pose, cac, 'CA', cac, 'C', beg, 'N ', f_cut_ang_cacn))
   cst_cut.append(addcst_ang(pose, cac, 'C ', beg, 'N', beg, 'CA', f_cut_ang_cnca))
   cst_cut.append(addcst_ang(pose, end, 'CA', end, 'C', nac, 'N ', f_cut_ang_cacn))
   cst_cut.append(addcst_ang(pose, end, 'C ', nac, 'N', nac, 'CA', f_cut_ang_cnca))
   if debug: print(f'minimize.py: score after chainbreak ang...... {sfxn(pose):10.3f}')
   # print(r.numeric.dihedral(
   #       pose.residue(cac).xyz('CA'),
   #       pose.residue(cac).xyz('C'),
   #       pose.residue(beg).xyz('N'),
   #       pose.residue(beg).xyz('CA'),
   #    ))
   cst_cut.append(addcst_dih(pose, cac, 'CA', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dih))
   cst_cut.append(addcst_dih(pose, end, 'CA', end, 'C', nac, 'N ', nac, 'CA', f_cut_dih))
   cst_cut.append(addcst_dih(pose, cac, 'O ', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dihO))
   cst_cut.append(addcst_dih(pose, end, 'O ', end, 'C', nac, 'N ', nac, 'CA', f_cut_dihO))
   if debug: print(f'minimize.py: score after chainbreak dihedral. {sfxn(pose):10.3f}')

   ############## metal constraints ################

   xyzorig = pose.residue(nresasym).xyz(metalname)
   # xyznew = xyzVec(xyzorig)
   znzncount = 0
   for ir in range(2 * nresasym, len(pose.residues), nresasym):
      if pose.residue(ir).is_virtual_residue(): break
      xyz = pose.residue(ir).xyz(metalname)
      if (xyzorig - xyz).length() < 3.0:
         # print('CONTACT ASYM ZN')
         # xyznew += xyz
         addcst_dis(pose, metalresno, metalname, ir, metalname, f_metal_olap)
         znzncount += 1
   assert znzncount is 3
   # xyznew[0] = xyznew[0] / 4
   # xyznew[1] = xyznew[1] / 4
   # xyznew[2] = xyznew[2] / 4
   # pose.set_xyz(AtomID(pose.residue(nresasym).atom_index(metalname), nresasym), xyznew)
   # pose.dump_pdb('foo.pdb')
   # assert 0

   if debug:
      print(f'minimize.py: score after metal olap ......... {sfxn(pose):10.3f}')

   allowed_elems = 'NOS'
   znpos = pose.residue(metalresno).xyz(1)
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
               if res.atom_name(ia) in (' OD2', ' OE2'):  # other COO O sometimes closeish
                  continue
               znbonded.append(aid)
   if len(znbonded) != metalnbonds:
      print('WRONG NO OF LIGANDING ATOMS', len(znbonded))
      if debug:
         for aid in znbonded:

            print(pose.residue(aid.rsd()).name(), pose.residue(aid.rsd()).atom_name(aid.atomno()))
            pose.dump_pdb('WRONG_NO_OF_LIGANDING_ATOMS.pdb')
            return None, None
         # raise ValueError(f'WRONG NO OF LIGANDING ATOMS {len(znbonded)}')

   # metal/lig distance constraints

   for i, aid in enumerate(znbonded):
      cst = rosetta.core.scoring.constraints.AtomPairConstraint(metalaid, aid, f_metal_lig_dist)
      cst_lig_dis.append(cst)
      pose.add_constraint(cst)

   if debug:
      print(f'minimize.py: score after metal dist ......... {sfxn(pose):10.3f}')
      for aid in znbonded:
         print(aid.rsd(), aid.atomno(),
               pose.residue(aid.rsd()).name(),
               pose.residue(aid.rsd()).atom_name(aid.atomno()))
   # assert 0

   # lig/metal/lig angle constraints (or dihedral in-place constraint for COO)
   # TODO kinda hacky... will need to be more general?

   for i, aid in enumerate(znbonded):
      ir, res = aid.rsd(), pose.residue(aid.rsd())
      if all(_ not in res.name() for _ in 'ASP CYS HIS GLU'.split()):
         return None, None
         assert 0, f'unrecognized res {res.name()}'
      if any(_ in res.name() for _ in 'ASP GLU'.split()):
         # metal comes off of OD1/OE1
         ir, coo = aid.rsd(), ('OD1 CG OD2' if 'ASP' in res.name() else 'OE1 CD OE2').split()
         cst_lig_ori.append(
            addcst_dih(pose, ir, coo[0], ir, coo[1], ir, coo[2], metalaid.rsd(), metalname,
                       f_metal_coo))
      else:
         if 'HIS' in res.name(): aname = 'HD1' if res.has('HD1') else 'HE2'
         if 'CYS' in res.name(): aname = 'HG'
         cst_lig_ori.append(
            addcst_ang(pose, ir, res.atom_name(aid.atomno()), metalaid.rsd(), metalname, ir,
                       aname, f_point_at_metal))

   if debug: print(f'minimize.py: score after metal dir........... {sfxn(pose):10.3f}')

   # for i, iaid in enumerate(znbonded):
   #    for j, jaid in enumerate(znbonded[:i]):
   #       # pripnt(i, j)
   #       cst = rosetta.core.scoring.constraints.AngleConstraint(iaid, metalaid, jaid, f_metal_lig_ang)
   #       cst_lig_ang.append(cst)
   #       pose.add_constraint(cst)

   if debug: print(f'minimize.py: score after lig angle added..... {sfxn(pose):10.3f}')

   ################ minimization #########################

   movemap = rosetta.core.kinematics.MoveMap()
   movemap.set_bb(True)
   movemap.set_chi(True)
   movemap.set_jump(False)
   # for i in allowed_jumps:
   # movemap.set_jump(True, i)
   minimizer = rosetta.protocols.minimization_packing.symmetry.SymMinMover(
      movemap, sfxn, 'lbfgs_armijo_nonmonotone', 0.001, True)  # tol, nblist
   # if sfxn.has_nonzero_weight(rosetta.core.scoring.ScoreType.cart_bonded):
   # minimizer.cartesian(True)
   minimizer.apply(pose)
   if debug:
      print(f'minimize.py: score after min no scale........ {sfxn(pose):10.3f}')
      pose.remove_constraints()
      print(f'minimize.py: score (no cst) after min no scale........ {sfxn(pose):10.3f}')

   # printscores(sfxn, pose)

   kw.timer.checkpoint(f'min scale 1.0')

   asym = rosetta.core.pose.Pose()
   rosetta.core.pose.symmetry.extract_asymmetric_unit(pose, asym, False)
   rosetta.core.pose.replace_pose_residue_copying_existing_coordinates(asym, metalresno, metalres)
   # asym.dump_pdb('asym.pdb')

   # for ir in metalresnos:
   #    rosetta.core.pose.replace_pose_residue_copying_existing_coordinates(pose, ir, metalres)
   # pose.dump_pdb('zafter.pdb')

   if debug: print(kw.timer)

   info = rp.Bunch()
   info.score = sfxn(pose)

   ############### score component stuff ################
   st = rosetta.core.scoring.ScoreType
   etot = pose.energies().total_energies()
   info.score_fa_atr = (etot[st.fa_atr])
   info.score_fa_rep = (etot[st.fa_rep])
   info.score_fa_sol = (etot[st.fa_sol])
   info.score_lk_ball = (etot[st.lk_ball] + etot[st.lk_ball_iso] + etot[st.lk_ball_bridge] +
                         etot[st.lk_ball_bridge_uncpl])
   info.score_fa_elec = (etot[st.fa_elec] + etot[st.fa_intra_elec])
   info.score_hbond_sr_bb = (etot[st.hbond_sr_bb] + etot[st.hbond_lr_bb] + etot[st.hbond_bb_sc] +
                             etot[st.hbond_sc])
   info.score_dslf_fa13 = (etot[st.dslf_fa13])
   info.score_atom_pair_constraint = (etot[st.atom_pair_constraint])
   info.score_angle_constraint = (etot[st.angle_constraint])
   info.score_dihedral_constraint = (etot[st.dihedral_constraint])
   info.score_omega = (etot[st.omega])
   info.score_rotamer = (etot[st.fa_dun] + etot[st.fa_dun_dev] + etot[st.fa_dun_rot] +
                         etot[st.fa_dun_semi] + etot[st.fa_intra_elec] + etot[st.fa_intra_rep] +
                         etot[st.fa_intra_atr_xover4] + etot[st.fa_intra_rep_xover4] +
                         etot[st.fa_intra_sol_xover4])
   info.score_ref = (etot[st.ref])
   info.score_rama_prepro = (etot[st.rama_prepro])
   info.score_cart_bonded = (etot[st.cart_bonded])
   info.score_gen_bonded = (etot[st.gen_bonded])

   ############### cst stuff ################

   pose.remove_constraints()
   info.score_wo_cst = sfxn(pose)
   # record_nonzero_energies(sfxn, pose)

   # for ir in range(1, pose.size() + 1):
   #    rep = pose.energies().residue_total_energies(ir)[st.fa_rep]
   #    if rep != 0: print(ir, 'rep', rep)
   # assert 0

   [pose.add_constraint(cst) for cst in cst_cut]
   info.score_cst_cut = sfxn(pose) - info.score_wo_cst
   pose.remove_constraints()

   [pose.add_constraint(cst) for cst in cst_lig_dis]
   [pose.add_constraint(cst) for cst in cst_lig_ang]
   [pose.add_constraint(cst) for cst in cst_lig_ori]
   info.score_cst_lig_ori = sfxn(pose) - info.score_wo_cst
   pose.remove_constraints()

   [pose.add_constraint(cst) for cst in cst_lig_dis]
   info.score_cst_lig_dis = sfxn(pose) - info.score_wo_cst
   pose.remove_constraints()

   [pose.add_constraint(cst) for cst in cst_lig_ang]
   info.score_cst_lig_ang = sfxn(pose) - info.score_wo_cst
   pose.remove_constraints()

   [pose.add_constraint(cst) for cst in cst_lig_ori]
   info.score_cst_lig_ori = sfxn(pose) - info.score_wo_cst
   pose.remove_constraints()

   return asym, info

def cluster_stuff():
   # cluster
   scores = np.array([r.info.score_fa_rep for r in results])
   sorder = np.argsort(scores)  # perm to sorted

   clustcen = np.arange(len(scores), dtype=np.int)
   if kw.cluster:
      print(f'{" CLUSTER ":#^100}')
      # crds sorted by score
      crd = np.stack([r.info.bbcoords for r in results])
      crd = crd[sorder].reshape(len(scores), -1)
      nbbatm = (results[0].asym_pose_min.size() - 1) * 3
      clustcen = rp.cluster.cookie_cutter(crd, kw.max_bb_redundancy * np.sqrt(nbbatm))
      clustcen = sorder[clustcen]  # back to original order
      results = [results[i] for i in clustcen]
      kw.timer.checkpoint('filter_redundancy')

      for iclust, r in enumerate(results):
         outdir = kw.output_prefix + os.path.dirname(r.info.tag)
         if not outdir: outdir = '.'
         clust_fname = outdir + '/clustered/clust%06i_' % iclust + os.path.basename(
            r.info.tag) + '.pdb'
         # print("CLUST DUMP", r.info.tag, clust_fname)
         r.asym_pose_min.dump_pdb(clust_fname)

   return

def addZN(pose, xyzmetal):
   # add zn
   znres = make_residue('VZN')
   pose.append_residue_by_jump(znres, 1)
   znresi = len(pose.residues)
   znpos = xyzVec(*xyzmetal[:3])
   zndelta = znpos - pose.residue(znresi).xyz(1)
   for ia in range(1, pose.residue(znresi).natoms() + 1):
      newxyz = zndelta + pose.residue(znresi).xyz(ia)
      pose.set_xyz(AtomID(ia, znresi), newxyz)

def postprocess(kw):
   # print(kw.inputs)
   results = list()
   for fn in kw.inputs:
      result = rp.load(fn)
      # print(result)
      # assert 0
      result.info.sequence = result.info.sequence.replace(':protein_cutpoint_upper', '')
      # print(result.seq)
      results.append(result)
   # print(f'loaded {len(results)} results')
   results = mof.result.results_to_xarray(results)
   # print(results.sequence)
   # print(len(results.sequence))
   results = results.sortby('score')

   # print(sum(results.score < 20) / len(results.score))
   # print(results)
   # print(results.bbcoords.shape)

   for seq, group in results.groupby('sequence'):
      tag = seq.replace(',ZN', '').replace(',', '-')
      # print(seq, len(group.score))
      coords = group.bbcoords.data.reshape(len(group.score), -1)
      nres = float(group.nres.data[0])
      # print
      clustcen, clustid = rp.cluster.cookie_cutter(coords, kw.max_bb_redundancy * nres)
      # print(np.sort(np.bincount(clustid))[-5:])
      # print(clustcen)
      # print(np.unique(clustcen))
      seqresults = group.sel(result=clustcen)
      seqresults.attrs.clear()

      # print(len(group.result), type(group), type(group.result), type(group.score),
      # type(group.pdb_fname))
      print(
         f'{tag:30} ngroup {len(group.score):4} ncoord {len(coords):4} nclust {len(clustcen):4} nuniq {len(np.unique(clustcen)):4} clustresult {len(seqresults.result):4}'
      )

      seqdir = 'cluster/' + tag + '/'
      os.makedirs(seqdir, exist_ok=True)
      rp.dump(seqresults, seqdir + tag + '_results.pickle')
      # print(group.pdb_fname[:10])
      # print('groupsize', len(group.score), len(group.result))
      # print('npdbs', len(seqresults.pdb_fname))

      for i, fn in enumerate(seqresults.pdb_fname.data):
         clustsize = np.sum(clustid == clustcen[i])
         origfname = kw.input_path + '/' + os.path.basename(fn)
         shutil.copyfile(origfname, seqdir + 'nb%04i_' % clustsize + os.path.basename(fn))

# works?
def _cluster_bb_dataset(ds):
   # scores = np.array([r.info.score_fa_rep for r in results])
   # sorder = np.argsort(scores)  # perm to sorted

   clustcen = np.arange(len(scores), dtype=np.int)

   print(f'{" CLUSTER ":#^100}')
   # crds sorted by score
   crd = np.stack([r.info.bbcoords for r in results])
   crd = crd[sorder].reshape(len(scores), -1)
   nbbatm = (results[0].asym_pose_min.size() - 1) * 3
   clustcen = rp.cluster.cookie_cutter(crd, kw.max_bb_redundancy * np.sqrt(nbbatm))
   clustcen = sorder[clustcen]  # back to original order
   results = [results[i] for i in clustcen]
   kw.timer.checkpoint('filter_redundancy')

   for iclust, r in enumerate(results):
      outdir = kw.output_prefix + os.path.dirname(r.info.tag)
      if not outdir: outdir = '.'
      clust_fname = outdir + '/clustered/clust%06i_' % iclust + os.path.basename(
         r.info.tag) + '.pdb'
      # print("CLUST DUMP", r.info.tag, clust_fname)
      r.asym_pose_min.dump_pdb(clust_fname)

   # print(kw.inputs)
   results = list()
   for fn in kw.inputs:
      print(fn)
      results += rp.load(fn)
   print(len(results))
