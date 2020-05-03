import mof, rpxdock as rp, numpy as np

def test_xtal_search_2res_i213(c3_peptide, rotcloud_asp, rotcloud_his):
   arg = rp.Bunch()
   arg.max_bb_redundancy = 1.0
   arg.err_tolerance = 1.5
   arg.dist_err_tolerance = 1.0
   arg.angle_err_tolerance = 15
   arg.min_dist_to_z_axis = 6.0
   arg.sym_axes_angle_tolerance = 5.0
   arg.angle_to_cart_err_ratio = 20.0
   arg.max_dun_score
   arg.min_cell_size = 0
   arg.max_cell_size = 999
   arg.clash_dis = 3.0
   arg.contact_dis = 7.0
   arg.min_contacts = 0
   arg.max_sym_score = 9e9
   arg.max_2res_score = 10

   search_spec = mof.xtal_search.XtalSearchSpec(
      spacegroup='i213',
      pept_orig=np.array([0, 0, 0, 1]),
      pept_axis=np.array([0, 0, 1, 0]),
      max_dun_score=arg.max_dun_score,
      sym_of_ligand=dict(HZ3='C3'),
      ligands=['HZ3'],
   )

   r = mof.xtal_search.xtal_search_two_residues(
      search_spec,
      c3_peptide,
      rotcloud_his,
      rotcloud_asp,
      **arg,
   )

   assert len(r) is 1
   assert np.allclose(r[0].xalign, [
      [-0.77574427, -0.25473024, 0.57735027, 6.70142185],
      [0.608475, -0.54444912, 0.57735027, 6.70142185],
      [0.16726927, 0.79917937, 0.57735027, 6.70142185],
      [0., 0., 0., 1.],
   ], atol=0.0001)

if __name__ == '__main__':
   test_xtal_search_2res_i213(
      mof.data.c3_peptide(),
      mof.data.rotcloud_asp_small(),
      mof.data.rotcloud_his_small(),
   )
