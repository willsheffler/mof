import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta import core
from mof import data

pyrosetta_flags = f'-mute all -output_virtual -extra_res_fa {data.HZ3_params} {data.HZ4_params} {data.HZD_params} -preserve_crystinfo'

pyrosetta.init(pyrosetta_flags)

scoretypes = core.scoring.ScoreType

chm = core.chemical.ChemicalManager.get_instance()
rts = chm.residue_type_set('fa_standard')

dun_sfxn = core.scoring.ScoreFunction()
dun_sfxn.set_weight(core.scoring.ScoreType.fa_dun, 1.0)

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
