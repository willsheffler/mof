import pyrosetta, numpy as np, rpxdock as rp, os
from functools import lru_cache
from pyrosetta import rosetta
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core import scoring
from mof import data

from pyrosetta.rosetta.numeric import xyzVector_double_t as rVec
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as rMat
from pyrosetta import AtomID
from pyrosetta.rosetta.core.scoring import ScoreType

################################################################

# pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta'

# pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta_cart'
pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} /home/sheffler/src/mof/mof/data/motifs/HZ4.params -preserve_crystinfo -renumber_pdb -beta_cart --dalphaball /home/sheffler/src/rosetta/master/source/external/DAlpahBall/DAlphaBall.gcc'

# pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta -output_virtual'

################################################################

pyrosetta.init(pyrosetta_flags)

chm = core.chemical.ChemicalManager.get_instance()
rts = chm.residue_type_set('fa_standard')

makelattice = lambda x: rosetta.protocols.cryst.MakeLatticeMover().apply(x)

def get_sfxn(tag='cli', **kw):
  kw = rp.Bunch(kw)
  tag = tag.lower()
  if tag is 'cli':
    return scoring.get_score_function()
  elif tag is 'rotamer':
    get_sfxn_weights(kw.sfxn_rotamer_weights)
    return get_sfxn_weights(kw.sfxn_rotamer_weights)
  elif tag is 'sterics':
    return get_sfxn_weights(kw.sfxn_sterics_weights)
  elif tag is 'minimize':
    return get_sfxn_weights(kw.sfxn_minimize_weights)
  else:
    return get_sfxn_weights(tag)
  raise ValueError(f'unknown rosetta scorefunction tag: {tag}')

def get_sfxn_weights(weights_file):
  wf = weights_file
  if not os.path.exists(wf):
    wf = os.path.join(data.weights_dir, wf)
    if not os.path.exists(wf):
      wf = os.path.join(data.weights_dir, wf + '.wts')
  if not os.path.exists(wf):
    raise ValueError(f'rosetta weights file does not exist: {weights_file}')
  return scoring.ScoreFunctionFactory.create_score_function(wf)

def printscores(sfxn, pose):
  for st in sfxn.get_nonzero_weighted_scoretypes():
    print(str(st)[10:], pose.energies().total_energies()[st])

def name2aid(pose, ires, aname):
  return AtomID(pose.residue(ires).atom_index(aname.strip()), ires)

def addcst_dis(pose, ires, iname, jres, jname, func):
  cst = rosetta.core.scoring.constraints.AtomPairConstraint(name2aid(pose, ires, iname),
                                                            name2aid(pose, jres, jname), func)
  pose.add_constraint(cst)
  return cst

def addcst_ang(pose, ires, iname, jres, jname, kres, kname, func):
  cst = rosetta.core.scoring.constraints.AngleConstraint(name2aid(pose, ires, iname),
                                                         name2aid(pose, jres, jname),
                                                         name2aid(pose, kres, kname), func)
  pose.add_constraint(cst)
  return cst

def addcst_dih(pose, ires, iname, jres, jname, kres, kname, lres, lname, func):
  cst = rosetta.core.scoring.constraints.DihedralConstraint(name2aid(pose, ires, iname),
                                                            name2aid(pose, jres, jname),
                                                            name2aid(pose, kres, kname),
                                                            name2aid(pose, lres, lname), func)
  pose.add_constraint(cst)
  return cst

def make_residue(resn):
  if len(resn) == 1:
    resn = aa123[resn]
  return core.conformation.ResidueFactory.create_residue(rts.name_map(resn))

def make_1res_pose(resn):
  res = make_residue(resn)
  pose = core.pose.Pose()
  pose.append_residue_by_jump(res, 0)
  return pose

def xform_pose(pose, xform):
  for ir in range(1, len(pose.residues) + 1):
    res = pose.residue(ir)
    for ia in range(1, res.natoms() + 1):
      old = res.xyz(ia)
      old = np.array([old[0], old[1], old[2], 1])
      new = xform @ old
      pose.set_xyz(AtomID(ia, ir), rVec(new[0], new[1], new[2]))

aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa123 = dict(
   A="ALA",
   C="CYS",
   D="ASP",
   E="GLU",
   F="PHE",
   G="GLY",
   H="HIS",
   I="ILE",
   K="LYS",
   L="LEU",
   M="MET",
   N="ASN",
   P="PRO",
   Q="GLN",
   R="ARG",
   S="SER",
   T="THR",
   V="VAL",
   W="TRP",
   Y="TYR",
)
aa321 = dict(
   ALA="A",
   CYS="C",
   ASP="D",
   GLU="E",
   PHE="F",
   GLY="G",
   HIS="H",
   ILE="I",
   LYS="K",
   LEU="L",
   MET="M",
   ASN="N",
   PRO="P",
   GLN="Q",
   ARG="R",
   SER="S",
   THR="T",
   VAL="V",
   TRP="W",
   TYR="Y",
)
