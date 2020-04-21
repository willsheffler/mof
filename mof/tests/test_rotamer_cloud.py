from mof.rotamer_cloud import *
from mof.pyrosetta_init import pyrosetta

def test_rotamer_cloud_his():
   chis = [np.arange(-180, 180, 3), np.arange(-180, 180, 8)]
   rotcloud = RotCloudHisZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (362, 2)
   assert rotcloud.rotframes.shape == (362, 4, 4)
   # rotcloud.dump_pdb('cloud_his.pdb')

def test_rotamer_cloud_dhis():
   chis = [np.arange(-180, 180, 3), np.arange(-180, 180, 8)]
   rotcloud = RotCloudDHisZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (362, 2)
   assert rotcloud.rotframes.shape == (362, 4, 4)
   # rotcloud.dump_pdb('cloud_dhis.pdb')

def test_rotamer_cloud_hisd():
   chis = [np.arange(-180, 180, 3), np.arange(-180, 180, 8)]
   rotcloud = RotCloudHisdZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (362, 2)
   assert rotcloud.rotframes.shape == (362, 4, 4)
   # rotcloud.dump_pdb('cloud_his.pdb')

def test_rotamer_cloud_dhisd():
   chis = [np.arange(-180, 180, 3), np.arange(-180, 180, 8)]
   rotcloud = RotCloudDHisdZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (362, 2)
   assert rotcloud.rotframes.shape == (362, 4, 4)
   # rotcloud.dump_pdb('cloud_his.pdb')

def test_rotamer_cloud_cys():
   chis = [np.arange(-180, 180, 6), np.arange(-180, 180, 8)]
   rotcloud = RotCloudCysZN(grid=chis, max_dun_score=4.0)
   assert rotcloud.rotchi.shape == (855, 2)
   assert rotcloud.rotframes.shape == (855, 4, 4)
   # rotcloud.dump_pdb('cloud_cys.pdb')

def test_rotamer_cloud_dcys():
   chis = [np.arange(-180, 180, 6), np.arange(-180, 180, 8)]
   rotcloud = RotCloudDCysZN(grid=chis, max_dun_score=4.0)
   assert rotcloud.rotchi.shape == (855, 2)
   assert rotcloud.rotframes.shape == (855, 4, 4)
   # rotcloud.dump_pdb('cloud_cys.pdb')

def test_rotamer_cloud_asp():
   chis = [np.arange(-180, 180, 8), np.arange(-180, 180, 5)]
   rotcloud = RotCloudAspZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (2054, 2)
   assert rotcloud.rotframes.shape == (2054, 4, 4)

   # rotcloud.dump_pdb('cloud_asp.pdb')
def test_rotamer_cloud_dasp():
   chis = [np.arange(-180, 180, 8), np.arange(-180, 180, 5)]
   rotcloud = RotCloudDAspZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (2054, 2)
   assert rotcloud.rotframes.shape == (2054, 4, 4)
   # rotcloud.dump_pdb('cloud_asp.pdb')

def test_rotamer_cloud_glu():
   chis = [np.arange(-180, 180, 6), np.arange(-180, 180, 12), np.arange(-180, 180, 6)]
   rotcloud = RotCloudGluZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (5746, 3)
   assert rotcloud.rotframes.shape == (5746, 4, 4)
   # rotcloud.dump_pdb('cloud_glu.pdb')

def test_rotamer_cloud_dglu():
   chis = [np.arange(-180, 180, 6), np.arange(-180, 180, 12), np.arange(-180, 180, 6)]
   rotcloud = RotCloudDGluZN(grid=chis, max_dun_score=5.0)
   assert rotcloud.rotchi.shape == (5746, 3)
   assert rotcloud.rotframes.shape == (5746, 4, 4)
   # rotcloud.dump_pdb('cloud_glu.pdb')

if __name__ == '__main__':
   test_rotamer_cloud_his()
   test_rotamer_cloud_dhis()
   test_rotamer_cloud_hisd()
   test_rotamer_cloud_dhisd()
   test_rotamer_cloud_cys()
   test_rotamer_cloud_dcys()
   test_rotamer_cloud_asp()
   test_rotamer_cloud_dasp()
   test_rotamer_cloud_glu()
   test_rotamer_cloud_dglu()
