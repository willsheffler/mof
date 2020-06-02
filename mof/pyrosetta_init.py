import pyrosetta, numpy as np
from pyrosetta import rosetta
from pyrosetta.rosetta import core
from mof import data

from pyrosetta.rosetta.numeric import xyzVector_double_t as rVec
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as rMat
from pyrosetta import AtomID

# pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta'
pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta_cart'
# pyrosetta_flags = f'-mute all -extra_res_fa {data.params.VZN} -preserve_crystinfo -renumber_pdb -beta -output_virtual'

pyrosetta.init(pyrosetta_flags)

scoretypes = core.scoring.ScoreType

chm = core.chemical.ChemicalManager.get_instance()
rts = chm.residue_type_set('fa_standard')

dun_sfxn = core.scoring.ScoreFunction()
dun_sfxn.set_weight(core.scoring.ScoreType.fa_dun, 1.0)

lj_sfxn = core.scoring.ScoreFunction()
lj_sfxn.set_weight(core.scoring.ScoreType.fa_atr, 1.0)
lj_sfxn.set_weight(core.scoring.ScoreType.fa_rep, 0.55)

makelattice = lambda x: rosetta.protocols.cryst.MakeLatticeMover().apply(x)

def printscores(sfxn, pose):
   for st in sfxn.get_nonzero_weighted_scoretypes():
      print(str(st)[10:], pose.energies().total_energies()[st])

def name2aid(pose, ires, aname):
   return AtomID(pose.residue(ires).atom_index(aname.strip()), ires)

def addcst_dis(pose, ires, iname, jres, jname, func):
   pose.add_constraint(
      rosetta.core.scoring.constraints.AtomPairConstraint(name2aid(pose, ires, iname),
                                                          name2aid(pose, jres, jname), func))

def addcst_ang(pose, ires, iname, jres, jname, kres, kname, func):
   pose.add_constraint(
      rosetta.core.scoring.constraints.AngleConstraint(name2aid(pose, ires, iname),
                                                       name2aid(pose, jres, jname),
                                                       name2aid(pose, kres, kname), func))

def addcst_dih(pose, ires, iname, jres, jname, kres, kname, lres, lname, func):
   pose.add_constraint(
      rosetta.core.scoring.constraints.DihedralConstraint(name2aid(pose, ires, iname),
                                                          name2aid(pose, jres, jname),
                                                          name2aid(pose, kres, kname),
                                                          name2aid(pose, lres, lname), func))

def get_res_energy(pose, st, ires):
   return

def get_dun_energy(pose, ires):
   dun_sfxn(pose)
   return pose.energies().residue_total_energies(ires)[scoretypes.fa_dun]

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
