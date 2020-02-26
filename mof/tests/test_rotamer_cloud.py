from mof.rotamer_cloud import *
from mof.pyrosetta_init import pyrosetta

def test_rotamer_cloud_basic():
   rotcloud = RotamerCloudHisZn()
   # print(rotcloud.rotframes.shape)
   # print(rotcloud.rotframes)
   assert rotcloud.rotframes.shape == (14, 4, 4)

if __name__ == '__main__':
   test_rotamer_cloud_basic()
