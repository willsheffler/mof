import numpy as np, rpxdock as rp
from mof import util
from mof.pyrosetta_init import make_1res_pose, get_dun_energy
from abc import ABC, abstractmethod

class RotamerCloud(ABC):
   """holds transforms for a set of rotamers positioned at the origin"""
   def __init__(
         self,
         amino_acid,
         rotchi=None,
         exchi=[30, 5],
         max_dun_score=4.0,
   ):
      super(RotamerCloud, self).__init__()
      self.amino_acid = amino_acid
      pose = make_1res_pose(amino_acid)
      if rotchi is None:
         rotchi = util.get_rotamers(pose.residue(1))
         rotchi = np.array([list(x) for x in rotchi])
         # print(rotchi)
         # print(rotchi.shape)
      assert len(rotchi), 'no chi angles specified'
      self.rotchi = rotchi
      self.rotorigin = _get_stub_1res(pose)
      rot_to_origin = np.linalg.inv(self.rotorigin)
      self.rotbin = []
      self.rotscore = []
      rotframes = []
      for irot, chis in enumerate(self.rotchi):
         for ichi, chi in enumerate(chis):
            pose.set_chi(ichi + 1, 1, chi)
         self.rotbin.append(irot)
         self.rotscore.append(get_dun_energy(pose, 1))
         endframe = self.get_effector_frame(pose.residue(1))
         rotframes.append(rot_to_origin @ endframe)

         # hacky test for only one specific case...rot_to_origin
         # this will fail in general, so comment it out
         assert np.allclose([x for x in pose.residue(1).xyz('VZN')],
                            (self.rotorigin @ rotframes[-1][:, 3])[:3])

      self.rotframes = np.stack(rotframes)
      self.pose1res = pose

   @abstractmethod
   def get_effector_frame(self, residue):
      pass

class RotamerCloudHisZn(RotamerCloud):
   def __init__(self, *args, **kw):
      super(RotamerCloudHisZn, self).__init__('HZD', *args, **kw)

   def get_effector_frame(self, residue):
      return rp.motif.frames.stub_from_points(
         residue.xyz('VZN'),
         residue.xyz('NE2'),
         residue.xyz('CE1'),
      ).squeeze()

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
