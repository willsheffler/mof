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
   f432=XtalSpecCD('F 4 3 2'),
   i4132=XtalSpecCD('I 41 3 2', ),
   # I4132 C2 0.707,0,0.707  -  -0.125,-0.125,0.125  D2 -0.707,0.707,0 0,0,1 0,-0.25,0.375  60 3.20E-17 -
   # I4132 C2 0.707,0,0.707  -  -0.125,-0.125,0.125  D2 -0.707,0.707,0 0.707,0.707,0  -0.5,-0.25,-0.375 45 0.125 -
   # I4132 C2 -0.707,0.707,0 -  0.375,0.375,0.375 D2 -0.707,0,0.707 0,1,0 -0.25,0.375,0  60 0.144338 -
   p23=XtalSpecCD('P 2 3'),
   p4232=XtalSpecCD('P 42 3 2'),

   # i432=XtalSpecCC('I 4 3 2', ),
   # I432  C2 0.707,0,0.707  -  -0.25,-0.25,0.25  D2 -0.707,0.707,0 0.707,0.707,0  -1,0,-0.5   45 0.25  -
   # I432  C2 0,0.707,0.707  -  -0.5,-0.5,0.5  D2 -0.707,0.707,0 0,0,1 0,-0.5,0.25 60 0.144338 -
   # I432  C2 0,0.707,0.707  -  -0.5,-0.5,0.5  D2 -0.707,0.707,0 0.707,0.707,0  0,-0.5,0.25 45 0.5   -
   # I432  C2 0,0.707,0.707  -  0,0,0 D2 -0.707,0.707,0 0.707,0.707,0  0,-0.5,-0.25   45 0  -
)
