from mof.rotamer_cloud import *
from mof.pyrosetta_init import pyrosetta

def test_rotamer_cloud_his():
   chis = [np.arange(-180, 180, 3), np.arange(-180, 180, 8), [-90]]
   # mesh = np.meshgrid(*chis, indexing='ij')
   # rotchi = np.stack(mesh, axis=len(mesh))
   # rotchi = rotchi.reshape(-1, 3)
   # rotcloud = RotamerCloudHisZn(rotchi=rotchi, max_dun_score=4.0)

   rotcloud = RotamerCloudHisZN(grid=chis, max_dun_score=5.0)

   assert rotcloud.rotchi.shape == (362, 3)
   assert rotcloud.rotframes.shape == (362, 4, 4)

   rotcloud.dump_pdb('cloud_his.pdb')

def test_rotamer_cloud_cys():

   chis = [np.arange(-180, 180, 6), np.arange(-180, 180, 8)]
   rotcloud = RotamerCloudCysZN(grid=chis, max_dun_score=4.0)

   assert rotcloud.rotchi.shape == (855, 2)
   assert rotcloud.rotframes.shape == (855, 4, 4)

   rotcloud.dump_pdb('cloud_cys.pdb')

def test_rotamer_cloud_asp():

   chis = [np.arange(-180, 180, 8), np.arange(-180, 180, 5)]
   rotcloud = RotamerCloudAspZN(grid=chis, max_dun_score=5.0)

   assert rotcloud.rotchi.shape == (158, 2)
   assert rotcloud.rotframes.shape == (158, 4, 4)

   rotcloud.dump_pdb('cloud_asp.pdb')

def test_rotamer_cloud_glu():

   chis = [np.arange(-180, 180, 5), np.arange(-180, 180, 10), np.arange(-180, 180, 5)]
   rotcloud = RotamerCloudGluZN(grid=chis, max_dun_score=6.0)

   assert rotcloud.rotchi.shape == (2642, 3)
   assert rotcloud.rotframes.shape == (2642, 4, 4)

   rotcloud.dump_pdb('cloud_glu.pdb')

if __name__ == '__main__':
   test_rotamer_cloud_his()
   test_rotamer_cloud_cys()
   test_rotamer_cloud_asp()
   test_rotamer_cloud_glu()
