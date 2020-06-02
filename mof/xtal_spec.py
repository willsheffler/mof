import numpy as np, rpxdock as rp
from rpxdock import homog as hm
from mof.data import data_dir

class XtalSpec:
   pass

class XtalSpecCC(XtalSpec):
   def __init__(
         self,
         spacegroup,
         nfold1,
         axis1,
         orig1,
         nfold2,
         axis2,
         orig2,
         nsubs,
   ):
      self.spacegroup = spacegroup
      self.nfold1 = int(nfold1)
      self.sym1 = 'C%i' % nfold1
      self.axis1 = hm.hnormalized(hm.hvec(axis1))
      self.orig1 = hm.hpoint(orig1)
      self.nfold2 = int(nfold2)
      self.sym2 = 'C%i' % nfold2
      self.axis2 = hm.hnormalized(hm.hvec(axis2))
      self.orig2 = hm.hpoint(orig2)
      self.nsubs = nsubs
      self.dihedral = np.degrees(hm.angle(axis1, axis2))
      self.axis1d = None
      self.axis2d = None
      try:
         # print(spacegroup, list(_frames_files.keys()))
         self.frames = rp.load(_frames_files[spacegroup])
         # print(self.frames.shape)
      except:
         self.frames = None

_frames_files = {
   'I 21 3': data_dir + '/i213_redundant111_n16_maxrad2.pickle',
   'P 41 3 2': data_dir + '/p4132_trionly_n12_maxrad3.pickle',
}

def get_xtal_spec(name):
   try:
      return _xspec[name]
   except KeyError:
      print(f'spacegroup {name} not implemented')

# _by_dihedral = dict()

_xspec = dict(
   f432=XtalSpecCC(
      'F 4 3 2',
      3,
      [1, 1, 1],
      [0.5, 0.5, 0],
      4,
      [1, 0, 0],
      [0, 0, 0],
      24,
   ),
   p213=XtalSpecCC(
      'P 21 3',
      3,
      [-1, +1, +1],
      [+0., +0, +0.5],
      3,
      [+1, +1, +1],
      [0, 0, 0],
      12,
   ),
   i213=XtalSpecCC(
      'I 21 3',
      3,
      [1, 1, 1],
      [0, 0, 0],
      2,
      [0, 0, 1],
      [0, -0.25, 0],
      12,
   ),
   p4132=XtalSpecCC(
      'P 41 3 2',
      3,
      [-1, -1, 1],
      [-0.5, 0, -0.5],
      2,
      [0, -1, 1],
      [-0.125, -0.125, -0.125],
      24,
   ),
   p4332=XtalSpecCC(
      'P 43 3 2',
      3,
      [-1, -1, 1],
      [-0.5, 0, -0.5],
      2,
      [0, -1, 1],
      [-0.375, -0.125, -0.125],
      24,
   ),

   # "I213  C2 0,0,1 -  0,-0.25,0   C3 0.57735,0.57735,0.57735 -  0,0,0 54.7356  0.176777 0",
   # "P4132 C2 0,-0.707107,0.707107 -  -0.125,-0.125,-0.125 C3 -0.57735,-0.57735,0.57735  -  -0.5,0,-0.5 35.2644  0.176777 -",
   # "P4332 C2 0,-0.707107,0.707107 -  -0.375,0.125,0.125   C3 0.57735,-0.57735,0.57735   -  0.333333,0.166667,-0.166667   35.2644  0.176777 -",
)

#    AXS = [Vec(1, 1, 1), Vec(1, 1, -1)]
#    CEN = [cell * Vec(0, 0, 0), cell * Vec(0.5, 0, 0.0)]

# _xspec = xspec_f432
