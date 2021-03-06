import numpy as np, rpxdock as rp, copy
from mof import util
from mof.pyrosetta_init import make_1res_pose, get_sfxn, rVec, xform_pose
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
      sfxn_rotamer = get_sfxn('rotamer')
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
      self.frameidx = list()
      self.rotframes = list()
      for irot, chis in enumerate(rotchi):
         for ichi, chi in enumerate(chis):
            pose.set_chi(ichi + 1, 1, chi)
         dun = sfxn_rotamer(pose)
         if dun > max_dun_score: continue
         # print('rot', irot, dun, chis)
         for iframe, frame in enumerate(self.get_effector_frame(pose.residue(1))):
            self.rotbin.append(irot)
            self.rotchi.append(chis)
            self.rotscore.append(dun)
            self.frameidx.append(iframe)
            self.rotframes.append(frame)

         # hacky test for only one specific case...self.to_origin
         # this will fail in general, so comment it out
         # assert np.allclose([x for x in pose.residue(1).xyz('VZN')],
         #      (self.origin @ self.rotframes[-1][:, 3])[:3])
      assert self.rotbin, 'no chi angles pass dun cut'

      self.rotbin = np.array(self.rotbin)
      self.rotscore = np.array(self.rotscore)
      self.rotchi = np.stack(self.rotchi)
      self.frameidx = np.stack(self.frameidx)
      self.rotframes = np.stack(self.rotframes)

      print(
         f'created RotamerCloud {self.amino_acid} nrots: {len(np.unique(self.rotbin))} nframes: {len(self.rotbin)}'
      )

   def make_pose1res(self):
      pose = make_1res_pose(self.amino_acid)
      xform_pose(pose, np.linalg.inv(self.original_origin))
      return pose

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
      res = self.make_pose1res().residue(1)
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

class RotCloudHisZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super().__init__('HIS', *args, **kw)

   def get_effector_frame(self, residue):
      cd = residue.xyz('CD2')
      ce = residue.xyz('CE1')
      ne = residue.xyz('NE2')

      zn = rVec(0, 0, 0)
      for i in range(3):
         zn[i] = ne[i] - (cd[i] + ce[i]) / 2
      zn.normalize()
      for i in range(3):
         zn[i] = ne[i] + 2.2 * zn[i]

      return [rp.motif.frames.stub_from_points(zn, ne, ce).squeeze()]

class RotCloudDHisZN(RotCloudHisZN):
   def __init__(self, *args, **kw):
      RotamerCloud.__init__(self, 'DHIS', *args, **kw)

class RotCloudHisdZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotCloudHisdZN, self).__init__('HIS_D', *args, **kw)

   def get_effector_frame(self, residue):
      cg = residue.xyz('CG')
      ce = residue.xyz('CE1')
      nd = residue.xyz('ND1')

      zn = rVec(0, 0, 0)
      for i in range(3):
         zn[i] = nd[i] - (cg[i] + ce[i]) / 2
      zn /= np.linalg.norm(zn)
      for i in range(3):
         zn[i] = nd[i] + 2.2 * zn[i]

      return [rp.motif.frames.stub_from_points(zn, nd, ce).squeeze()]

class RotCloudDHisdZN(RotCloudHisdZN):
   def __init__(self, *args, **kw):
      RotamerCloud.__init__(self, 'DHIS_D', *args, **kw)

class RotCloudCysZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotCloudCysZN, self).__init__('CYS', *args, **kw)

   def get_effector_frame(self, residue):
      hg = residue.xyz('HG')
      sg = residue.xyz('SG')
      cb = residue.xyz('CB')
      orig = (hg - sg).normalized()
      for i in range(3):
         orig[i] = orig[i] * 2.32 + sg[i]
      frame = rp.motif.frames.stub_from_points(orig, sg, cb).squeeze()
      return [frame]

class RotCloudDCysZN(RotCloudCysZN):
   def __init__(self, *args, **kw):
      RotamerCloud.__init__(self, 'DCYS', *args, **kw)

class RotCloudAspZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotCloudAspZN, self).__init__('ASP', *args, **kw)

   def get_effector_frame(self, residue):
      return _asp_glu_effectors(residue)

class RotCloudDAspZN(RotCloudAspZN):
   def __init__(self, *args, **kw):
      RotamerCloud.__init__(self, 'DASP', *args, **kw)

class RotCloudGluZN(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotCloudGluZN, self).__init__('GLU', *args, **kw)

   def get_effector_frame(self, residue):
      return _asp_glu_effectors(residue)

class RotCloudDGluZN(RotCloudGluZN):
   def __init__(self, *args, **kw):
      RotamerCloud.__init__(self, 'DGLU', *args, **kw)

def _asp_glu_effectors(residue):
   if residue.name() in ('ASP', 'DASP'):
      names = 'CG', 'OD1', 'OD2'
   elif residue.name() in ('GLU', 'DGLU'):
      names = 'CD', 'OE1', 'OE2'
   else:
      raise NotImplementedError

   c = np.array(residue.xyz(names[0])).reshape(1, 3)
   o1 = np.array(residue.xyz(names[1])).reshape(1, 3)
   o2 = np.array(residue.xyz(names[2])).reshape(1, 3)

   # print(rp.homog.angle_degrees(o1 - c, o2 - c))
   # print(rp.homog.angle_degrees(o1 - o2, o1 - c))
   # print('----')

   orig = (c - o2) / np.linalg.norm(c - o2)
   orig = o1 + orig * 2.1

   c = rp.homog.hpoint(c)
   o1 = rp.homog.hpoint(o1)
   o2 = rp.homog.hpoint(o2)
   orig = rp.homog.hpoint(orig)
   rotaxis = rp.homog.hcross(o1 - c, o2 - c)
   rot = rp.homog.hrot(rotaxis, 10, o1, degrees=True).squeeze()
   frames = list()
   for irot in range(13):
      # print(
      #    rp.homog.angle_degrees(orig - o1, o1 - c),
      #    rp.homog.angle_degrees(orig - o1, c - o2),
      # )
      frame = rp.motif.frames.stub_from_points(orig, o1, c).squeeze()
      frames.append(frame)
      # print(rot.shape, orig.shape)
      orig = (rot @ orig.squeeze()).reshape(1, 4)
   return frames

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

# for pickle test file compatibility... this is very lazy... should regen
RotamerCloudAspZN = RotCloudAspZN
RotamerCloudCysZN = RotCloudCysZN
RotamerCloudGluZN = RotCloudGluZN
RotamerCloudHisZN = RotCloudHisZN
