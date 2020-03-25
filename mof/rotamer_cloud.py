import numpy as np, rpxdock as rp, copy
from mof import util
from mof.pyrosetta_init import make_1res_pose, get_dun_energy, rVec, xform_pose
from abc import ABC, abstractmethod
"""
CONCERNS:

how to handle multiple metal binding sides not covered by rotamers, as in GLU

how to hangle CYS chi2, which is based on HG being free-ish to rotate
"""

class RotamerCloud(ABC):
   """holds transforms for a set of rotamers positioned at the origin"""
   def __init__(
         self,
         amino_acid,
         rotchi=None,
         max_dun_score=4.0,
         grid=None,
   ):
      super(RotamerCloud, self).__init__()
      self.amino_acid = amino_acid
      pose = make_1res_pose(amino_acid)
      if rotchi is None:
         if grid is None:
            rotchi = util.get_rotamers(pose.residue(1))
            rotchi = np.array([list(x) for x in rotchi])
         else:
            mesh = np.meshgrid(*grid, indexing='ij')
            rotchi = np.stack(mesh, axis=len(mesh))
            rotchi = rotchi.reshape(-1, len(mesh))

      # print(rotchi)
      # print(rotchi.shape)
      assert len(rotchi), 'no chi angles specified'
      self.original_rotchi = rotchi
      self.original_origin = _get_stub_1res(pose)
      xform_pose(pose, np.linalg.inv(self.original_origin))
      self.rotchi = list()
      self.rotbin = list()
      self.rotscore = list()
      rotframes = list()
      for irot, chis in enumerate(rotchi):
         for ichi, chi in enumerate(chis):
            pose.set_chi(ichi + 1, 1, chi)
         dun = get_dun_energy(pose, 1)
         if dun > max_dun_score: continue
         # print('rot', irot, dun, chis)
         self.rotbin.append(irot)
         self.rotchi.append(chis)
         self.rotscore.append(dun)
         rotframes.append(self.get_effector_frame(pose.residue(1)))

         # hacky test for only one specific case...self.to_origin
         # this will fail in general, so comment it out
         # assert np.allclose([x for x in pose.residue(1).xyz('VZN')],
         #      (self.origin @ rotframes[-1][:, 3])[:3])
      assert self.rotbin, 'no chi angles pass dun cut'

      self.rotbin = np.array(self.rotbin)
      self.rotscore = np.array(self.rotscore)
      self.rotchi = np.stack(self.rotchi)
      self.rotframes = np.stack(rotframes)
      self.pose1res = pose

      print('RotamerCloud', self.amino_acid, self.rotchi.shape)

   def subset(self, which):
      new_one = copy.copy(self)
      new_one.rotchi = self.rotchi[which]
      new_one.rotbin = self.rotbin[which]
      new_one.rotscore = self.rotscore[which]
      new_one.rotframes = self.rotframes[which]
      return new_one

   @abstractmethod
   def get_effector_frame(self, residue):
      pass

   def dump_pdb(self, path=None, position=np.eye(4), which=None):
      if path is None: path = self.amino_acid + '.pdb'
      res = self.pose1res.residue(1)
      natm = res.natoms()
      F = rp.io.pdb_format_atom
      with open(path, 'w') as out:
         loopey_doodle = enumerate(self.rotchi)
         if which is not None:
            loopey_doodle = ((which, self.rotchi[which]), )
         for irot, chis in loopey_doodle:
            out.write('MODEL %i\n' % irot)
            for ichi, chi in enumerate(chis):
               res.set_chi(ichi + 1, chi)
            for ia in range(1, natm + 1):
               xyz = res.xyz(ia)
               xyz = position @ np.array([xyz[0], xyz[1], xyz[2], 1])
               line = F(ia=ia, ir=1, an=res.atom_name(ia), rn=res.name3(), c='A', xyz=xyz)
               out.write(line)
            orig = self.rotframes[irot, :, 3]
            x = orig + 2 * self.rotframes[irot, :, 0]
            y = orig + 2 * self.rotframes[irot, :, 1]
            z = orig + 2 * self.rotframes[irot, :, 2]
            orig = position @ orig
            x = position @ x
            y = position @ y
            z = position @ z
            # print(self.rotframes[irot])
            # print(orig)
            # print(x)
            # print(y)
            # print(z)
            # assert 0
            out.write(F(ia=natm + 1, ir=1, an='ORIG', rn='END', c='B', xyz=orig))
            out.write(F(ia=natm + 2, ir=1, an='XDIR', rn='END', c='B', xyz=x, elem='O'))
            out.write(F(ia=natm + 3, ir=1, an='YDIR', rn='END', c='B', xyz=y, elem='CL'))
            out.write(F(ia=natm + 4, ir=1, an='ZDIR', rn='END', c='B', xyz=z, elem='N'))
            out.write('ENDMDL\n')

   def __len__(self):
      return len(self.rotchi)

# def pdb_format_atom(ia=0, an="ATOM", idx=" ", rn="RES", c="A", ir=0, insert=" ", x=0, y=0, z=0,
# occ=1, b=1, elem=" ", xyz=None):

class RotamerCloudHisZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotamerCloudHisZN, self).__init__('HZD', *args, **kw)

   def get_effector_frame(self, residue):
      return rp.motif.frames.stub_from_points(
         residue.xyz('VZN'),
         residue.xyz('NE2'),
         residue.xyz('CE1'),
      ).squeeze()

class RotamerCloudCysZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotamerCloudCysZN, self).__init__('CYS', *args, **kw)

   def get_effector_frame(self, residue):
      hg = residue.xyz('HG')
      sg = residue.xyz('SG')
      cb = residue.xyz('CB')
      orig = (hg - sg).normalized()
      for i in range(3):
         orig[i] = orig[i] * 2.32 + sg[i]
      return rp.motif.frames.stub_from_points(orig, sg, cb).squeeze()

class RotamerCloudAspZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotamerCloudAspZN, self).__init__('ASP', *args, **kw)

   def get_effector_frame(self, residue):
      cg = residue.xyz('CG')
      od1 = residue.xyz('OD1')
      od2 = residue.xyz('OD2')
      orig = (od1 - od2).normalized()
      for i in range(3):
         orig[i] = orig[i] * 2.1 + od1[i]
      return rp.motif.frames.stub_from_points(orig, od1, cg).squeeze()

class RotamerCloudGluZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotamerCloudGluZN, self).__init__('GLU', *args, **kw)

   def get_effector_frame(self, residue):
      cd = residue.xyz('CD')
      oe1 = residue.xyz('OE1')
      oe2 = residue.xyz('OE2')
      orig = (oe1 - oe2).normalized()
      for i in range(3):
         orig[i] = orig[i] * 1.83 + oe1[i]
      return rp.motif.frames.stub_from_points(orig, oe1, cd).squeeze()

def _get_stub_1res(pose):
   res = pose.residue(1)
   n = res.xyz('N')
   ca = res.xyz('CA')
   c = res.xyz('C')
   return rp.motif.frames.bb_stubs(
      np.array([[n[0], n[1], n[2]]]),
      np.array([[ca[0], ca[1], ca[2]]]),
      np.array([[c[0], c[1], c[2]]]),
   ).squeeze()
