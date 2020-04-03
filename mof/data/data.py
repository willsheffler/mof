import os, pyrosetta, logging, _pickle

log = logging.getLogger(__name__)

data_dir = os.path.dirname(__file__)

frank_space_groups = os.path.join(data_dir, "crystals_from_point.csv")

peptides_dir = os.path.join(data_dir, "peptides")
a_c3_peptide = os.path.join(peptides_dir, "c3_21res_c.101.12_0001.pdb")

motifs_dir = str(os.path.join(data_dir, "motifs"))
HZ3_params = str(os.path.join(motifs_dir, "HZ3.params"))
HZD_params = str(os.path.join(motifs_dir, "HZD.params"))
HZ4_params = str(os.path.join(motifs_dir, "HZ4.params"))
all_params_files = " ".join([HZ3_params, HZ4_params, HZD_params])

def c3_peptide():
   return pyrosetta.pose_from_file(os.path.join(data_dir, 'peptides/c3_21res_c.107.7_0001.pdb'))

def rotcloud_asp_small():
   return load(os.path.join(data_dir, 'rotcloud_asp_small.pickle'))

def rotcloud_his_small():
   return load(os.path.join(data_dir, 'rotcloud_his_small.pickle'))

def load(f, verbose=True):
   if isinstance(f, str):
      if verbose: log.debug(f'loading{f}')
      with open(f, "rb") as inp:
         return _pickle.load(inp)
   return [load(x) for x in f]

def dump(thing, f):
   d = os.path.dirname(f)
   if d: os.makedirs(d, exist_ok=True)
   with open(f, "wb") as out:
      return _pickle.dump(thing, out)
