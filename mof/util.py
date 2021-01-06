import sys, pandas, numpy as np, rmsd, rpxdock as rp, rpxdock.homog as hm, xarray as xr
from hashlib import sha1
import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec, xyzMatrix_double_t as xyzMat
from pyrosetta.bindings.utility import bind_method
from pyrosetta.rosetta.core.pack.rotamer_set import bb_independent_rotamers
from pyrosetta.rosetta.core.conformation import Residue

def extra_rotamers(rotamers, lb=0, ub=1, bs=10):
   ex = []
   for rotamer in rotamers:
      for i in range(lb, ub, bs):
         for j in range(lb, ub, bs):
            ex.append([rotamer[1] + i, rotamer[2] + j, rotamer[3]])
   return ex

def show_body_isect(body, Xalign, maxdis=3.0):
   body2 = body.copy()
   body2.move_to(Xalign)
   assert not np.allclose(body.pos, body2.pos, atol=1e-4)
   pairs = body.contact_pairs(body2, maxdis=3.0, atomno=True)
   assert len(pairs)
   # print(pairs.shape)
   # print('contact_pairs\n', pairs[:10])
   pdb1, pdb2 = list(), list()
   for i, j in pairs:
      crd1 = body.positioned_coord_atomno(i)
      crd2 = body2.positioned_coord_atomno(j)
      pdb1.append(rp.io.pdb_format_atom(xyz=crd1))
      pdb2.append(rp.io.pdb_format_atom(xyz=crd2))
      # print(i, j, np.linalg.norm(crd1 - crd2))
   print('dumping clash1/2.pdb')
   rp.util.dump_str(pdb1, 'clash1.pdb')
   rp.util.dump_str(pdb2, 'clash2.pdb')
   body.dump_pdb('clash1_body1.pdb')
   body2.dump_pdb('clash2_body2.pdb')

# def get_xtal_info():
#    df = pandas.read_csv(data.frank_space_groups)
#    sgdat = dict()
#    for key in [
#          'spacegroup/cage', 'sym_type_1', 'sym_type_2', 'dihedral', 'offset',
#          'shift(angle_0_only)'
#    ]:
#       sgdat[key] = df[key]
#
#    goodrows = np.ones(len(df['offset']), dtype='bool')
#    for key in [
#          'sym_axis_1', 'sym_axis_1D(D_only)', 'origin_1', 'sym_axis_2', 'sym_axis_2D(D_only)',
#          'origin_2'
#    ]:
#       dat = list()
#       for i, val in enumerate(df[key]):
#          raw = val.split(',')
#          if raw[0] == '-':
#             if not key.count('D_only'):
#                goodrows[i] = 0
#             dat.append([-12345] * 4)
#             continue
#          if len(raw) is 1: raw = [raw[0]] * 3
#          homog = 1.0 if key in 'origin_1 origin_2'.split() else 0.0
#          dat.append([float(x) for x in raw] + [homog])
#       sgdat[key] = np.array(dat)
#
#    for k, v in sgdat.items():
#       goodcol = v[goodrows]
#       if v.ndim > 1:
#          sgdat[k] = (['spacegroup', 'xyzw'], goodcol)
#       else:
#          sgdat[k] = (['spacegroup'], goodcol)
#
#    ds = xr.Dataset(sgdat)
#
#    # print(ds)
#    # print(ds['sym_type_2'])
#    # print(np.sum(ds['sym_type_2'] == 'C3'))
#
#    return ds

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

def prep_poses(pose_gen):
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
   for path, pose in pose_gen:
      mut_d.set_selector(checkable_res_pos)
      mut_d.set_res_name('DALA')
      mut_d.apply(pose)
      mut_l.set_selector(checkable_res_neg)
      mut_l.set_res_name('ALA')
      mut_l.apply(pose)
      center_pose(pose)
      yield path, pose
   #    prepped_pdbs.append(pose)
   # return prepped_pdbs

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

def fix_bb_h(pose, ires):
   # if pose.residue(ires).name() == 'PRO': return
   # if pose.residue(ires).name() == 'DPRO': return
   c = np.array(pose.residue((ires - 2) % len(pose.residues) + 1).xyz('C'))
   n = np.array(pose.residue(ires).xyz('N'))
   ca = np.array(pose.residue(ires).xyz('CA'))

   hpos = (n - (ca + c) / 2)
   hpos = n + hpos / np.linalg.norm(hpos)

   hpos = xyzVec(*hpos[:3])
   if pose.residue(ires).has('H'):
      pose.set_xyz(AtomID(pose.residue(ires).atom_index('H'), ires), hpos)

# def fix_bb_o(pose, ires):
#    r = pose.residue(ires)
#    io = r.atom_index('O')
#    crd = r.build_atom_ideal(io, pose.conformation())
#    pose.set_xyz(AtomID(io, ires), crd)

def center_pose(pose):
   crds = list()
   for i in range(1, len(pose.residues) + 1):
      crds.append(np.array(pose.residue(i).xyz('CA')))
   crds = np.stack(crds)
   tocen = -np.mean(crds, axis=0)
   # print(tocen)
   Rx = rosetta.numeric.xyzMatrix_double_t.cols(1, 0, 0, 0, 1, 0, 0, 0, 1)
   v = rosetta.numeric.xyzVector_double_t(tocen[0], tocen[1], tocen[2])
   pose.apply_transform_Rx_plus_v(Rx, v)

def fix_bb_h_all(pose):
   for ir in range(1, len(pose.residues) + 1):
      fix_bb_h(pose, ir)

def mutate_two_res(pose, ires1in, aa1, chis1, ires2in, aa2, chis2, symnum=1):
   pose2 = pose.clone()
   assert len(pose2.residues) % symnum == 0
   nres = len(pose2.residues) // symnum
   for isym in range(symnum):
      ires1 = (ires1in - 1) % nres + 1 + isym * nres
      ires2 = (ires2in - 1) % nres + 1 + isym * nres
      mut = rosetta.protocols.simple_moves.MutateResidue()
      mut.set_preserve_atom_coords(False)
      mut.set_res_name(aa1)
      mut.set_target(ires1)
      mut.apply(pose2)
      for ichi, chi in enumerate(chis1):
         pose2.set_chi(ichi + 1, ires1, chi)
      mut.set_res_name(aa2)
      mut.set_target(ires2)
      mut.apply(pose2)
      for ichi, chi in enumerate(chis2):
         pose2.set_chi(ichi + 1, ires2, chi)
      # pose2.dump_pdb('before.pdb')
      for ir in range(1, len(pose2.residues) + 1):
         pose2.set_xyz(AtomID(pose2.residue(ir).atom_index('O'), ir), pose.residue(ir).xyz('O'))
         if pose2.residue(ir).name3() != 'PRO':
            fix_bb_h(pose2, ir)
      # pose2.dump_pdb('after.pdb')
   return pose2

def mutate_one_res(pose, ires1in, aa1, chis1, symnum=1):
   pose2 = pose.clone()
   assert len(pose2.residues) % symnum == 0
   nres = len(pose2.residues) // symnum
   for isym in range(symnum):
      ires1 = (ires1in - 1) % nres + 1 + isym * nres
      mut = rosetta.protocols.simple_moves.MutateResidue()
      mut.set_preserve_atom_coords(False)
      mut.set_res_name(aa1)
      mut.set_target(ires1)
      mut.apply(pose2)
      for ichi, chi in enumerate(chis1):
         pose2.set_chi(ichi + 1, ires1, chi)
      for ir in range(1, len(pose2.residues) + 1):
         pose2.set_xyz(AtomID(pose2.residue(ir).atom_index('O'), ir), pose.residue(ir).xyz('O'))
         if pose2.residue(ir).name3() != 'PRO':
            fix_bb_h(pose2, ir)
      # pose2.dump_pdb('after.pdb')
   return pose2

def mut_to_ligand(pose, residue, ligands, sym_of_ligand, debug=False):
   # For a given pose, attempts to mutate each residue (one at a time) to a new residue from my list of ligands
   # Returns a dictionary, LIG_poses, that contains all of the new poses with mutated residue positions
   # where "a" = poses, "b" = interaction residue (HZ3, HZ4, etc.)
   sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function('ref2015')
   LIG_poses = []
   int_residues = []
   lig_sym = []

   for ilig in range(1, len(ligands), 2):
      ligand = ligands[ilig]

      LIG_pose = pose.clone()
      if LIG_pose.phi(residue) > 0:
         mut = rosetta.protocols.simple_moves.MutateResidue()
         mut.set_res_name(ligand)
         mut.set_target(residue)
         mut.set_preserve_atom_coords(False)
         # print(ligand, residue)
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
      lig_sym.append(sym_of_ligand[ligand])
      if debug:
         print(LIG_poses)
         print(LIG_pose.residue_type(residue))
      zipbObj = zip(LIG_poses, zip(int_residues, lig_sym))
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
   rosrots = bb_independent_rotamers(self.type())
   if self.type().is_d_aa():
      rotamers = [pyrosetta.Vector1([-chi for chi in rotamer.chi()]) for rotamer in rosrots]
   else:
      rotamers = [rotamer.chi() for rotamer in rosrots]
   return rotamers

# @bind_method(Residue)
# def set_random_rotamer(self):
#    # Randomly selects a rotamer for finding rotamers
#    rotamers = self.get_rotamers()
#    one_rot = rotamers[random.randint(0, len(rotamers))]
#    if debug:
#       print("-----this is the random rotamer selected", one_rot, "out of total", len(rotamers))
#    for i in range(1, len(one_rot) + 1):  # Sets the random rotamer
#       self.set_chi(i, one_rot[i])

# def align_HIS(orig, rotd, resi, r_end):
#    # orig is the original position you want to get aligned to
#    # rotd is short for rotated and is the pose that is being rotated
#    # resi is the residue number in the orig poses that we are aligning
#    # and r_end is the residue number in the rotd pose
#    p_scaff = []
#    p_targ = []
#    for atom in range(6, orig.residue(resi).natoms() + 1):
#       #if debug:
#       #print ("orig resi is: ", orig.residue(resi))
#       #print ("atom number of orig resi is: ", atom)
#       if (not orig.residue(resi).atom_is_hydrogen(atom)):
#          p_scaff.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))
#          #if debug:
#          #print ("p_scaff is: ", p_scaff)
#          p_targ.append(coord_find(orig, resi, orig.residue(resi).atom_name(atom)))
#
#    #step1: moving scaffold to the center
#    T = find_cent(p_scaff)
#    plusv = rosetta.numeric.xyzVector_double_t(-1 * T[0], -1 * T[1], -1 * T[2])
#    #does not rotate
#    noR = rosetta.numeric.xyzMatrix_double_t.cols(1, 0, 0, 0, 1, 0, 0, 0, 1)
#    rotd.apply_transform_Rx_plus_v(noR, plusv)
#
#    #Step1': get the coordinates of target at the center
#    T_targ = find_cent(p_targ)
#    v_targ = rosetta.numeric.xyzVector_double_t(-1 * T_targ[0], -1 * T_targ[1], -1 * T_targ[2])
#    orig.apply_transform_Rx_plus_v(noR, v_targ)
#
#    #need to re-load the matrix now because the pose has changed
#    p_scaff_new = []
#    p_targ_new = []
#    for atom in range(6, orig.residue(resi).natoms() + 1):
#       #if (not (orig.residue(resi).atom_is_backbone(atom))):
#       if (not orig.residue(resi).atom_is_hydrogen(atom)):
#          p_scaff_new.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))
#          p_targ_new.append(coord_find(orig, resi, orig.residue(resi).atom_name(atom)))
#
#    #Step 2: get the rotation matrix
#    #the magic of libraries
#    #V=rmsd.kabsch(p_targ_new,p_scaff_new)
#    semi_V = rmsd.kabsch(p_scaff_new, p_targ_new)
#    V = np.linalg.inv(semi_V)
#
#    #Rotate the pose
#    Rx = rosetta.numeric.xyzMatrix_double_t.cols(V[0][0], V[1][0], V[2][0], V[0][1], V[1][1],
#                                                 V[2][1], V[0][2], V[1][2], V[2][2])
#    noT = rosetta.numeric.xyzVector_double_t(0, 0, 0)
#
#    #moving the pose
#    rotd.apply_transform_Rx_plus_v(Rx, noT)
#
#    #Step3: translate the pose back to target (both the new and the original)
#    scaff_trans = rosetta.numeric.xyzVector_double_t(T_targ[0], T_targ[1], T_targ[2])
#    rotd.apply_transform_Rx_plus_v(noR, scaff_trans)
#    orig.apply_transform_Rx_plus_v(noR, scaff_trans)
#
#    #generating final set
#    p_scaff_final = []
#    for atom in range(6, orig.residue(resi).natoms() + 1):
#       #if (not (orig.residue(resi).atom_is_backbone(atom))):
#       if (not orig.residue(resi).atom_is_hydrogen(atom)):
#          p_scaff_final.append(coord_find(rotd, r_end, rotd.residue(r_end).atom_name(atom)))
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

# def magic_angle(pose, residue_number, debug=False):
#
#    p1 = coord_find(pose, residue_number, 'VZN')
#    p2 = coord_find(pose, residue_number, 'HZ')
#    p1_a = np.array(p1)
#    p2_a = np.array(p2)
#
#    peptide_axis = [0, 0, 1]
#    interaction_axis = p1_a - p2_a
#    m_a = (angle(peptide_axis, interaction_axis) * (180 / np.pi))
#
#    if debug:
#       print("P1:", p1)
#       print("P2:", p2)
#       print("P1_A", p1_a)
#       print("P2_A", p2_a)
#       print("INTERACTION_AXIS:", interaction_axis)
#
#    return m_a
"""change xyz to coordinates"""

def coord_find(p, ir, ia):

   coord_xyz = p.xyz(rosetta.core.id.AtomID(p.residue(ir).atom_index(ia), ir))
   coord_arr = [coord_xyz[0], coord_xyz[1], coord_xyz[2]]
   return np.array(coord_arr)

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
   return np.sqrt(dotproduct(v, v))

def angle(v1, v2):
   if dotproduct(v1, v2) == 1:
      return 0
   return np.arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def output_data(pdb_name, residue, mut_res_name, pose_num, list_of_spacegroups):
   file_name = '{}_{}_{}_{}.txt'.format(pdb_name, residue, mut_res_name, pose_num)
   o = open(file_name, "w")
   o.write("PDB NAME: %s\n" % pdb_name)
   o.write("MUTATED RESIDUE NUMBER: %s\n" % residue)
   o.write("MUTATED RESIDUE ID: %s\n" % mut_res_name)
   o.write("ROTAMER NUMBER: %s\n\n" % pose_num)
   o.write("COMPATIBLE SPACEGROUP(S): \n%s\n" % list_of_spacegroups)

def hash_str_to_int(s):
   if isinstance(s, str):
      s = s.encode()
   buf = sha1(s).digest()[:8]
   return int(abs(np.frombuffer(buf, dtype="i8")[0]))
