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
      if hm.angle(self.axis1, self.axis2) > np.pi / 2:
         self.axis2[:3] = -self.axis2[:3]
      assert 90 > hm.angle_degrees(self.axis1, self.axis2)
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

class XtalSpecCD(XtalSpec):
   def __init__(
      self,
      spacegroup,
      nfold1,
      axis1,
      orig1,
      nfold2,
      axis2,
      orig2,
      axis2d,
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
      if hm.angle(self.axis1, self.axis2) > np.pi / 2:
         self.axis2[:3] = -self.axis2[:3]
      assert 90 > hm.angle_degrees(self.axis1, self.axis2)

      self.orig2 = hm.hpoint(orig2)
      self.nsubs = nsubs
      self.dihedral = np.degrees(hm.angle(axis1, axis2))
      self.axis1d = None
      self.axis2d = hm.hnormalized(hm.hvec(axis2d))
      try:
         # print(spacegroup, list(_frames_files.keys()))
         self.frames = rp.load(_frames_files[spacegroup])
         # print(self.frames.shape)
      except:
         self.frames = None

_frames_files = {
   'I 21 3': data_dir + '/i213_redundant111_n16_maxrad2.pickle',
   'P 41 3 2': data_dir + '/p4132_trionly_n12_maxrad3.pickle',
   'P 43 3 2': data_dir + '/p4132_trionly_n12_maxrad3.pickle',
}

def get_xtal_spec(name):
   try:
      return _xspec[name.lower()]
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
      [+1, +1, +1],
      [0, 0, 0],
      3,
      [-1, +1, +1],
      [+0., +0, +0.5],
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
   # i213=XtalSpecCC(
   #    'I 21 3',
   #    2,
   #    [0, 1, 0],
   #    [-0.25, 0, -0.5],
   #    3,
   #    [-1, 1, 1],
   #    [-1 / 6, 1 / 6, -1 / 3],
   #    12,
   # ),
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
   p23=XtalSpecCD(
      'P 2 3',
      3,
      [1, -1, 1],
      [-2 / 3, -1 / 3, 1 / 3],
      2,
      [0, 1, 0],
      [-1 / 2, -1 / 2, 0],
      [1, 0, 0],
      12,
   ))

# tetrahedral breaks my symframe generation grr....

# F432  C3 0.57735,0.57735,0.57735      0.333333,-0.166667,-0.166667     D2 0.707107,0,0.7071   -0.707107,0,0.707107      0.25,-0.5,-0.25   54.7356  1.70E-17
# F432  C3 0.57735,0.57735,0.57735      0.333333,-0.166667,-0.166667     D2 0,0,1 -0.707107,0.707107,0                    0.25,-0.25,-0.5   35.2644  8.01E-18
# F432  C3 0.57735,0.57735,0.57735      0.333333,-0.166667,-0.166667     D2 0,1,0 0.707107,0,0.707107                    -0.5,-0.5,-0.5 90 0.204124
# F432  C3 0.57735,0.57735,0.57735      -0.333333,-0.333333,0.666667     D2 0,-0.707107,0.707107 1,0,0                   -0.5,-0.25,-0.25  35.2644  0.707107
# F432  C3 0.57735,0.57735,0.57735      -0.5,0,0.5                       D2 -0.707107,0.707107,0 0.707107,0.707107,0     -1,0,-0.5   54.7356  0.353553
# I4132 C3 0.57735,-0.57735,0.57735        0,-0.5,-0.5                   D2 1,0,0              0,-0.707107,0.707107      0.125,-0.5,-0.25  90 1.39E-17
# I4132 C3 0.57735,-0.57735,0.57735        0,-0.5,-0.5                   D2 -0.707107,0.707107,0     0.707107,0.707107,0 -0.5,-0.25,-0.375 54.7356  0.176777
# I4132 C3 0.57735,-0.57735,0.57735        0,-0.5,-0.5                   D2 0,0,1 0.707107,0.707107,0                    -0.5,-0.25,0.125  35.2644  0.176777
# I4132 C3 0.57735,0.57735,0.57735      -0.333333,-0.333333,0.666667     D2 0,0,1             0.707107,0.707107,0        -0.5,-0.25,0.125  90 0.408248

# P23   C3 0.57735,-0.57735,0.57735        -0.666667,-0.333333,0.333333  D2 0,1,0                   1,0,0                  -0.5,-0.5,0 54.7356  3.40E-17

# P23   C3 -0.57735,-0.57735,0.57735       0,0,0                         D2 0,0,1 0,1,0 -0.5,-0.5,0 54.7356  0.353553
# P4232 C3 -0.57735,-0.57735,0.57735       0,0,0                         D2 -0.707107,0.707107,0 0.707107,0.707107,0    -0.5,0,-0.25   54.7356  0.353553
# P4232 C3 -0.57735,-0.57735,0.57735       0,0,0                         D2 -0.707107,0.707107,0 0,0,1                  -0.5,0,-0.25   35.2644  0.353553
# P4232 C3 -0.57735,-0.57735,0.57735       0,0,0                         D2 0,-0.707107,0.707107 1,0,0                  -0.25,0,-0.5   90 0.408248
