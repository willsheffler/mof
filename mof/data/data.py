import os
data_dir = os.path.dirname(__file__)

frank_space_groups = os.path.join(data_dir, "crystals_from_point.csv")

peptides_dir = os.path.join(data_dir, "peptides")
a_c3_peptide = os.path.join(peptides_dir, "c3_21res_c.101.12_0001.pdb")

motifs_dir = str(os.path.join(data_dir, "motifs"))
HZ3_params = str(os.path.join(motifs_dir, "HZ3.params"))
HZD_params = str(os.path.join(motifs_dir, "HZD.params"))
HZ4_params = str(os.path.join(motifs_dir, "HZ4.params"))
all_params_files = " ".join([HZ3_params, HZ4_params, HZD_params])
