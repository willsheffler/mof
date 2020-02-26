import numpy as np

from pyrosetta.rosetta.core.import_pose import pose_from_file
from mof.tests import path_to_test_data
from mof import util, pyrosetta_init, xtal_spec
from mof.xtal_build import xtal_build
import rpxdock as rp

def test_xtal_build_p213():
   pose = pose_from_file(path_to_test_data('test_xtal_build_p213.pdb'))

   xtal_posess = xtal_build(
      pdb_name='testpdb',
      xspec=xtal_spec.get_xtal_spec('p213'),
      pose=pose,
      ires=1,
      peptide_sym='C3',
      peptide_orig=np.array([0, 0, 0, 1]),
      peptide_axis=np.array([0, 0, 1, 0]),
      metal_sym='C3',
      metal_origin=np.array([6.21626192, -9.70902624, 5.32956683, 1.]),
      metal_sym_axis=np.array([-0.55345815, -0.76326468, -0.33333333, 0.]),
      rpxbody=rp.Body(pose),
      tag='test_xtal_build_p213',
   )

   xalign, xpose, bodypdb = xtal_posess[0]
   print('verify bb coords')
   assert np.allclose(
      util.coord_find(xpose, 1, 'CA'),
      [-2.33104726, -4.58076386, -11.17135244],
   )
   assert np.allclose(
      util.coord_find(xpose, 2, 'CA'),
      [-5.24447638, -2.71997589, -12.75316055],
   )
   assert np.allclose(
      util.coord_find(xpose, 3, 'CA'),
      [-5.3331455, -4.53568805, -16.09169707],
   )
   print('verify space group and unit cell')
   assert xpose.pdb_info().crystinfo().spacegroup() == 'P 21 3'
   assert np.allclose(xpose.pdb_info().crystinfo().A(), 30.354578479691444)

   # print('test_xtal_build_p213')

   print('verify xtal alignment')
   xalign_should_be = np.array([
      [-0.47930882, -0.6610066, 0.57735027, -7.73090447],
      [0.81210292, -0.08459031, 0.57735027, -7.73090447],
      [-0.3327941, 0.74559691, 0.57735027, -7.73090447],
      [0., 0., 0., 1.],
   ])
   assert np.allclose(xalign, xalign_should_be, atol=1e-5)

if __name__ == '__main__':
   test_xtal_build_p213()