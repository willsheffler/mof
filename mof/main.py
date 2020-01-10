import pyrosetta
from pyrosetta import rosetta

import pandas
import numpy as np
import math

from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.core.pack.rotamer_set import bb_independent_rotamers
from pyrosetta.rosetta.core.conformation import Residue
from pyrosetta.rosetta.core.pose import Pose

import rmsd

pyrosetta.init('-extra_res_fa motifs/HZ4.params motifs/HZ3.params')
_DEBUG = False
leeway = 1

def main():
   # imports poses - requires a "pdbs.list" file containing paths
   raw_pdbs = read_in_pdbs()

   r  # preps poses - makes all non-PRO (l&d) non-AIB residues into Alanines
   prepped_pdbs = prep_poses(raw_pdbs)

   # read FD & UN's materials info into a pandas DataFrame (df)
   df = pandas.read_csv('crystals_from_point.csv')
   df1 = df[[sym_axis in x for x in df['sym_type_1']]]
   df2 = df[[sym_axis in x for x in df['sym_type_2']]]
   # this df contains only applicable crystal space groups/cage types for a given scaffold
   df_sym = df1.append(df2)

   chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
   rts = chm.residue_type_set('fa_standard')
   scfxn = rosetta.core.scoring.ScoreFunction()
   scfxn.set_weight(rosetta.core.scoring.ScoreType.fa_dun, 1.0)

   for pdb in prepped_pdbs:
      # gets the pdb name for outputs later
      p_n = pdb.pdb_info().name().split('/')[-1]
      # gets rid of the ".pdb" at the end of the pdb name
      pdb_name = p_n[:-4]

      # check the symmetry type of the pdb
      last_res = rosetta.core.pose.chain_end_res(pdb).pop()
      total_res = int(last_res)
      sym_num = pdb.chain(last_res)
      sym = int(sym_num)
      sym_axis = str(sym_num)

      for residue in range(1, int(total_res / sym) + 1):

         if pdb.residue_type(residue) == rts.name_map('ALA') or pdb.residue_type(
               residue) == rts.name_map('DALA'):
            # one residue at a time, mutate to a residue that is involved in one of
            # the defined interactions. Stores the output in a dictionary w/
            #  "a" = poses, "b" = interaction residue (his, asp, cys, etc.)
            LIG_poses = mut_to_ligand(pdb, residue)

            for LIG_pose in LIG_poses:
               transform(LIG_pose, [1, 0, 0])
               # gets all of the backbone independent rotamers
               rotamers = LIG_pose.residue(residue).get_rotamers()

               pose_num = 1
               for rotamer in rotamers:
                  #for i in range(1, len(rotamer)+1): # if I want to sample the metal-axis too
                  for i in range(1, len(rotamer) + 1):
                     LIG_pose.residue(residue).set_chi(i, rotamer[i])
                  ROT_pose = rosetta.protocols.grafting.return_region(
                     LIG_pose, 1, LIG_pose.size())
                  if _DEBUG:
                     ROT_pose.dump_pdb('CHECK1_{}_{}_{}.pdb'.format(residue, LIG_poses[LIG_pose],
                                                                    pose_num))

                  scfxn(ROT_pose)
                  dun_score = ROT_pose.energies().residue_total_energies(residue)[
                     rosetta.core.scoring.ScoreType.fa_dun]

                  if dun_score < 3:  # scores the mutated residue's fa_dun to only get "good" rotamers (bb_dependent)
                     list_of_spacegroups = []
                     if _DEBUG:
                        ROT_pose.dump_pdb('CHECK2_{}_{}_{}.pdb'.format(
                           residue, LIG_poses[LIG_pose], pose_num))

                     # if this dihedral is compatible with any of the spacegroups in the dataframe (df)
                     # based on its calculated_dihedral and a pre-set "leeway" value I've set above,
                     # it'll add the spacegroup to my "list_of_spacegroups" to store the info.
                     calculated_dihedral = magic_angle(ROT_pose, residue)

                     if _DEBUG:
                        print("CALCULATED DIHEDRAL:", calculated_dihedral, "POSE_NUM", pose_num)

                     condition = ((df_sym['dihedral'] > (calculated_dihedral - leeway)) &
                                  (df_sym['dihedral'] < (calculated_dihedral + leeway)))
                     list_of_spacegroups.append(
                        (df_sym['spacegroup/cage'].where(condition)).dropna())
                     mut_res_name = LIG_poses[LIG_pose]
                     str_los = str(list_of_spacegroups)
                     if len(str_los) > 50:
                        output_data(pdb_name, residue, mut_res_name, pose_num,
                                    list_of_spacegroups)
                        ROT_pose.dump_pdb('successful_{}_{}_{}_{}.pdb'.format(
                           pdb_name, residue, LIG_poses[LIG_pose], pose_num))
                  else:
                     print("This is a bad rotamer:", dun_score)
                     continue

                  pose_num += 1
         else:
            continue

if __name__ == '__main__':
   main()

### defs ###
def read_in_pdbs():
   # Reads a pdbs.list file to get the paths to all of the pdbs I want to test,
   # then opens the pdbs with import_pose.pose_from_file
   # Returns a list, raw_pose_list, that contains all of the poses
   raw_pose_path_list = []
   raw_pose_list = []
   pdbslist = open("pdbs.list", "r")
   lines = pdbslist.read().splitlines()
   for line in lines:
      raw_pose_path_list.append(line)
   for path in raw_pose_path_list:
      raw_pose = rosetta.core.import_pose.pose_from_file(path)
      raw_pose_list.append(raw_pose)
   return raw_pose_list

def variant_remove(pose):
   # Takes in a pose, if there are any variant types on it, it gets rid of it
   # This makes it so that the atom number dependent functions later on don't get
   # tripped up by the extra atoms that get added on if you have a terminus variant
   for res in range(1, pose.size() + 1):
      if (pose.residue(res).has_variant_type(
            pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT)):
         pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
            pose, pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT, res)
      if (pose.residue(res).has_variant_type(
            pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT)):
         pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
            pose, pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT, res)
      if (pose.residue(res).has_variant_type(pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER)):
         pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
            pose, pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER, res)
      if (pose.residue(res).has_variant_type(pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER)):
         pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
            pose, pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER, res)

def cyc_align(pose):
   # Takes in a pose, cyclizes it, then aligns it to the z-axis
   # Returns a pose
   pcm = rosetta.protocols.cyclic_peptide.PeptideCyclizeMover()
   ata = rosetta.protocols.cyclic_peptide.SymmetricCycpepAlign()
   pcm.apply(pose)
   ata.apply(pose)
   return pose

def prep_poses(pose_list):
   # Does the same thing as cyc_align, but for a list of poses
   # ALSO: makes all non-PRO (l&d) non-AIB residues into Alanines
   # Returns a new list of poses, preped_pdbs
   create_residue = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue
   chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
   rts = chm.residue_type_set('fa_standard')
   ala_inst = rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('ALA'))
   dala_inst = rosetta.core.conformation.ResidueFactory.create_residue(rts.name_map('DALA'))

   AIB_Pp = rosetta.core.select.residue_selector.ResidueNameSelector('PRO,DPR,AIB', True)
   pos_phi = rosetta.core.select.residue_selector.PhiSelector()
   pos_phi.set_select_positive_phi(True)  # True=select +phi
   neg_phi = rosetta.core.select.residue_selector.NotResidueSelector(pos_phi)
   rtc = rosetta.core.select.residue_selector.NotResidueSelector(AIB_Pp)
   checkable_res_pos = rosetta.core.select.residue_selector.AndResidueSelector(rtc, pos_phi)
   checkable_res_neg = rosetta.core.select.residue_selector.AndResidueSelector(rtc, neg_phi)

   prepped_pdbs = []
   mut_d = rosetta.protocols.simple_moves.MutateResidue()
   mut_l = rosetta.protocols.simple_moves.MutateResidue()
   for pose in pose_list:
      mut_d.set_selector(checkable_res_pos)
      mut_d.set_res_name('DALA')
      mut_d.apply(pose)
      mut_l.set_selector(checkable_res_neg)
      mut_l.set_res_name('ALA')
      mut_l.apply(pose)
      prepped_pdbs.append(pose)
   return prepped_pdbs

def minimize(pose, sfxn):
   # Minimizes the input pose using the input scorefunction
   movemap = rosetta.core.kinematics.MoveMap()
   movemap.set_chi(True)
   movemap.set_bb(False)
   movemap.set_jump(False)
   minmover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
      movemap, sfxn, 'linmin', 0.001, True)
   for i in range(5):
      minmover.apply(pose)

def mut_to_ligand(pose, residue):
   # For a given pose, attempts to mutate each residue (one at a time) to a new residue from my list of ligands
   # Returns a dictionary, LIG_poses, that contains all of the new poses with mutated residue positions
   # where "a" = poses, "b" = interaction residue (HZ3, HZ4, etc.)
   sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function('ref2015')
   LIG_poses = []
   int_residues = []
   ligands = ['HZ3', 'DHZ3', 'HZ4', 'DHZ4']

   for x in range(0, 1 + 1):
      ligand = ligands[2 * x + 1]
      LIG_pose = pose.clone()
      if LIG_pose.phi(residue) > 0:
         mut = rosetta.protocols.simple_moves.MutateResidue()
         mut.set_res_name(ligand)
         mut.set_target(residue)
         mut.set_preserve_atom_coords(False)
         mut.apply(LIG_pose)
      else:
         mut = rosetta.protocols.simple_moves.MutateResidue()
         mut.set_res_name(ligand)
         mut.set_target(residue)
         mut.set_preserve_atom_coords(False)
         mut.apply(LIG_pose)
      minimize(LIG_pose, sfxn)
      LIG_poses.append(LIG_pose)
      int_residues.append(
         str(LIG_pose.residue_type(residue)).partition('\n')[0].partition(' ')[0])
      if _DEBUG:
         print(LIG_poses)
         print(LIG_pose.residue_type(residue))
      zipbObj = zip(LIG_poses, int_residues)
      dict_pose_res = dict(zipbObj)
   return dict_pose_res

def transform(pose, target_v):
   # Calculates the transformations necessary to move the input pose
   # from its original position to [0,0,1]
   originial_axis = [0, 0, 1]  # this must be a unit vector
   cross_p = np.cross(originial_axis, target_v)
   dot_p = np.dot(originial_axis, target_v)
   skew_symmetric_matrix = np.matrix([[0, -cross_p[2], cross_p[1]], [cross_p[2], 0, -cross_p[0]],
                                      [-cross_p[1], cross_p[0], 0]])
   I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
   '''V is rotation matrix'''
   A = I + skew_symmetric_matrix + skew_symmetric_matrix * skew_symmetric_matrix * (1 /
                                                                                    (1 + dot_p))
   # rosetta does not understand matrices, so we need to spoon feed it an array
   V = np.squeeze(np.asarray(A))
   Rx = rosetta.numeric.xyzMatrix_double_t.cols(V[0][0], V[1][0], V[2][0], V[0][1], V[1][1],
                                                V[2][1], V[0][2], V[1][2], V[2][2])
   noT = rosetta.numeric.xyzVector_double_t(0, 0, 0)
   pose.apply_transform_Rx_plus_v(Rx, noT)

@bind_method(Residue)
def get_rotamers(self):
   # Gets rotamers for a given residue
   if self.type().is_d_aa():
      rotamers = [
         pyrosetta.Vector1([-chi
                            for chi in rotamer.chi()])
         for rotamer in bb_independent_rotamers(self.type())
      ]
   else:
      rotamers = [rotamer.chi() for rotamer in bb_independent_rotamers(self.type())]
   return rotamers

@bind_method(Residue)
def set_random_rotamer(self):
   # Randomly selects a rotamer for finding rotamers
   rotamers = self.get_rotamers()
   one_rot = rotamers[random.randint(0, len(rotamers))]
   if _DEBUG:
      print("-----this is the random rotamer selected", one_rot, "out of total", len(rotamers))
   for i in range(1, len(one_rot) + 1):  # Sets the random rotamer
      self.set_chi(i, one_rot[i])

def align_HIS(orig, rotd, resi, r_end):
   # orig is the original position you want to get aligned to
   # rotd is short for rotated and is the pose that is being rotated
   # resi is the residue number in the orig poses that we are aligning
   # and r_end is the residue number in the rotd pose
   p_scaff = []
   p_targ = []
   for atom in range(6, orig.residue(resi).natoms() + 1):
      #if _DEBUG:
      #print ("orig resi is: ", orig.residue(resi))
      #print ("atom number of orig resi is: ", atom)
      if (not orig.residue(resi).atom_is_hydrogen(atom)):
         p_scaff.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))
         #if _DEBUG:
         #print ("p_scaff is: ", p_scaff)
         p_targ.append(coord_find(orig, resi, orig.residue(resi).atom_name(atom)))

   #step1: moving scaffold to the center
   T = find_cent(p_scaff)
   plusv = rosetta.numeric.xyzVector_double_t(-1 * T[0], -1 * T[1], -1 * T[2])
   #does not rotate
   noR = rosetta.numeric.xyzMatrix_double_t.cols(1, 0, 0, 0, 1, 0, 0, 0, 1)
   rotd.apply_transform_Rx_plus_v(noR, plusv)

   #Step1': get the coordinates of target at the center
   T_targ = find_cent(p_targ)
   v_targ = rosetta.numeric.xyzVector_double_t(-1 * T_targ[0], -1 * T_targ[1], -1 * T_targ[2])
   orig.apply_transform_Rx_plus_v(noR, v_targ)

   #need to re-load the matrix now because the pose has changed
   p_scaff_new = []
   p_targ_new = []
   for atom in range(6, orig.residue(resi).natoms() + 1):
      #if (not (orig.residue(resi).atom_is_backbone(atom))):
      if (not orig.residue(resi).atom_is_hydrogen(atom)):
         p_scaff_new.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))
         p_targ_new.append(coord_find(orig, resi, orig.residue(resi).atom_name(atom)))

   #Step 2: get the rotation matrix
   #the magic of libraries
   #V=rmsd.kabsch(p_targ_new,p_scaff_new)
   semi_V = rmsd.kabsch(p_scaff_new, p_targ_new)
   V = np.linalg.inv(semi_V)

   #Rotate the pose
   Rx = rosetta.numeric.xyzMatrix_double_t.cols(V[0][0], V[1][0], V[2][0], V[0][1], V[1][1],
                                                V[2][1], V[0][2], V[1][2], V[2][2])
   noT = rosetta.numeric.xyzVector_double_t(0, 0, 0)

   #moving the pose
   rotd.apply_transform_Rx_plus_v(Rx, noT)

   #Step3: translate the pose back to target (both the new and the original)
   scaff_trans = rosetta.numeric.xyzVector_double_t(T_targ[0], T_targ[1], T_targ[2])
   rotd.apply_transform_Rx_plus_v(noR, scaff_trans)
   orig.apply_transform_Rx_plus_v(noR, scaff_trans)

   #generating final set
   p_scaff_final = []
   for atom in range(6, orig.residue(resi).natoms() + 1):
      #if (not (orig.residue(resi).atom_is_backbone(atom))):
      if (not orig.residue(resi).atom_is_hydrogen(atom)):
         p_scaff_final.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))

"""find axis function for metal_ligand"""

def plane_normal_finder(point1, point2, point3=[]):

   if len(point3) > 0:
      p1 = np.array(point1)
      p2 = np.array(point2)
      p3 = np.array(point3)

      # These two vectors are in the plane
      v1 = p3 - p1
      v2 = p2 - p1

      # the cross product is a vector normal to the plane
      cp = np.cross(v1, v2)
      a, b, c = cp

      return cp

def magic_angle(pose, residue_number):

   p1 = coord_find(pose, residue_number, 'VZN')
   p2 = coord_find(pose, residue_number, 'HZ')
   p1_a = np.array(p1)
   p2_a = np.array(p2)

   peptide_axis = [1, 0, 0]
   interaction_axis = p1_a - p2_a
   m_a = (angle(peptide_axis, interaction_axis) * (180 / np.pi))

   if _DEBUG:
      print("P1:", p1)
      print("P2:", p2)
      print("P1_A", p1_a)
      print("P2_A", p2_a)
      print("INTERACTION_AXIS:", interaction_axis)

   return m_a

"""change xyz to coordinates"""

def coord_find(p, ir, ia):

   coord_xyz = p.xyz(rosetta.core.id.AtomID(p.residue(ir).atom_index(ia), ir))
   coord_arr = []
   x = coord_xyz[0]
   y = coord_xyz[1]
   z = coord_xyz[2]
   coord_arr.append(x)
   coord_arr.append(y)
   coord_arr.append(z)

   return coord_arr

"""getting the center of several points"""

def find_cent(A):

   sumA = [0, 0, 0]
   for i in range(len(A)):
      sumA[0] = sumA[0] + A[i][0]
      sumA[1] = sumA[1] + A[i][1]
      sumA[2] = sumA[2] + A[i][2]

   for i in range(3):
      sumA[i] = sumA[i] / len(A)

   return sumA

"""get the angle between two vectors"""

def dotproduct(v1, v2):
   return sum((a * b) for a, b in zip(v1, v2))

def length(v):
   return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
   if dotproduct(v1, v2) == 1:
      return 0
   return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def output_data(pdb_name, residue, mut_res_name, pose_num, list_of_spacegroups):
   file_name = '{}_{}_{}_{}.txt'.format(pdb_name, residue, mut_res_name, pose_num)
   o = open(file_name, "w")
   o.write("PDB NAME: %s\n" % pdb_name)
   o.write("MUTATED RESIDUE NUMBER: %s\n" % residue)
   o.write("MUTATED RESIDUE ID: %s\n" % mut_res_name)
   o.write("ROTAMER NUMBER: %s\n\n" % pose_num)
   o.write("COMPATIBLE SPACEGROUP(S): \n%s\n" % list_of_spacegroups)