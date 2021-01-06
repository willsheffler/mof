import mof, rpxdock as rp, numpy as np, os, rpxdock.homog as hm
from mof.pyrosetta_init import rosetta, xform_pose, make_residue
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta import AtomID
import pyrosetta

def main_loop_c3d2():

   for i in range(3):
      print('sbtdnbawiethli3q4jgyluahieorsvbnrasdsenh')
   os.system('rm test_*.pdb')

   kw = mof.options.get_cli_args()
   kw.timer = rp.Timer().start()
   if len(kw.inputs) is 0:
      kw = get_test_kw(kw)
   # for k, v in kw.items():
   #    try:
   #       print(k, v)
   #    except ValueError:
   #       print(k, type(v))
   rotclouds = mof.rotamer_cloud.get_rotclouds(**kw)

   pept_axis = np.array([0, 0, 1, 0])
   tetrahedral_angle = 109.47122063449069
   stepdegrees = 3

   count = 0

   for path in kw.inputs:

      pose = rosetta.core.import_pose.pose_from_file(path)
      rpxbody = rp.Body(pose)
      sym_num = pose.chain(len(pose.residues))
      nresasym = len(pose.residues) // sym_num

      for spacegroup in kw.spacegroups:
         search_spec = mof.xtal_search.XtalSearchSpec(
            spacegroup=spacegroup,
            pept_orig=np.array([0, 0, 0, 1]),
            pept_axis=np.array([0, 0, 1, 0]),
            # are these necessary:
            sym_of_ligand=dict(HZ3='C3', DHZ3='C3', HZ4='C4', DHZ4='C4', HZD='D2', DHZD='D2'),
            ligands=['HZ3', 'DHZ3'],
            **kw,
         )
         xspec = search_spec.xtal_spec
         target_angle = hm.line_angle(xspec.axis1, xspec.axis2)

         for iaa, aa in enumerate(kw.aa_labels):

            rotcloud = rotclouds[_lblmap[aa]]

            for ires in range(nresasym):

               print(f'{f"{spacegroup} {aa} {ires}":*^80}')
               stub = rpxbody.stub[ires]

               rotframes = stub @ rotcloud.rotframes

               for irot, rotframe in enumerate(rotframes):
                  lig_metal_axis = rotframe[:, 0]
                  symaxis0 = hm.hrot(rotframe[:, 1], tetrahedral_angle / 2) @ lig_metal_axis
                  symaxis0d = hm.hrot(lig_metal_axis, 120.0) @ symaxis0
                  symaxis0d2 = hm.hrot(lig_metal_axis, 240.0) @ symaxis0

                  rots_around_nezn = hm.xform_around_dof_for_vector_target_angle(
                     fix=pept_axis,
                     mov=symaxis0,
                     dof=lig_metal_axis,
                     target_angle=target_angle,
                  )

                  for ibonddof, x in enumerate(rots_around_nezn):
                     symaxis = x @ symaxis0
                     symaxisd = x @ symaxis0d
                     symaxisd2 = x @ symaxis0d2
                     if (np.pi / 2 < hm.angle(pept_axis, symaxis)):
                        symaxis *= -1

                     # print('oaiernst', ibonddof)
                     # print('pept_axis', pept_axis)
                     # print('symaxis', symaxis)
                     # print(hm.angle_degrees(pept_axis, symaxis))
                     # print('xspec.axis1', xspec.axis1)
                     # print('xspec.axis2', xspec.axis2)
                     # print(hm.angle_degrees(xspec.axis1, xspec.axis2))
                     # assert 0

                     # compute xalign

                     def line_plane_intesection(p0, n, l0, l):
                        l = hm.hnormalized(l)
                        d = hm.hdot(p0 - l0, n) / hm.hdot(l, n)
                        return l0 + l * d

                     assert np.allclose(hm.angle_degrees(pept_axis, symaxis),
                                        hm.angle_degrees(xspec.axis1, xspec.axis2), atol=0.01)

                     xalign = hm.align_vectors(pept_axis, symaxis, xspec.axis1, xspec.axis2)

                     # print(xalign @ pept_axis, xspec.axis1)
                     # print(xalign @ symaxis, xspec.axis2)
                     assert np.allclose(xalign @ pept_axis, xspec.axis1, atol=0.001)
                     assert np.allclose(xalign @ symaxis, xspec.axis2, atol=0.001)
                     orig_metal_pos = rotframe[:, 3]

                     xalign[:3, 3] = -(xalign @ orig_metal_pos)[:3]
                     assert np.allclose(xalign @ orig_metal_pos, [0, 0, 0, 1])
                     # xalign[:3, 3] += 5 * xspec.orig2[:3]
                     # print(xspec.orig2)
                     l = hm.hnormalized(hm.hvec(xspec.orig2))
                     l0 = hm.hpoint([0, 0, 0])
                     n = hm.hcross(xspec.axis1, xspec.orig1)
                     # p0 = xalign[:, 3]
                     p0 = xalign @ [0, 0, 0, 1]  # center of peptide
                     isect = line_plane_intesection(p0, n, l0, l)
                     xalign[:3, 3] -= isect[:3]  # why minus? huh

                     axis1_orig = hm.line_line_closest_points_pa(
                        [0, 0, 0, 1],
                        hm.hvec(xspec.orig1),
                        xalign[:, 3],
                        xspec.axis1,
                     )
                     axis1_orig = (axis1_orig[0] + axis1_orig[1]) / 2
                     axis2_orig = xalign @ orig_metal_pos
                     # print(hm.hnorm(axis1_orig), axis1_orig)
                     # print(hm.hnorm(axis2_orig), axis2_orig)
                     # print(axis1_orig / xspec.orig1)
                     # print(axis2_orig / xspec.orig2)

                     for n, d in zip(axis1_orig[:3], xspec.orig1[:3]):
                        if d != 0:
                           cell_spacing1 = n / d
                     for n, d in zip(axis2_orig[:3], xspec.orig2[:3]):
                        if d != 0:
                           cell_spacing2 = n / d
                     cellerr = (abs(cell_spacing1 - cell_spacing2) /
                                (abs(cell_spacing1) + abs(cell_spacing2)))
                     print('cell_spacing', cell_spacing1, cell_spacing2, cellerr)
                     if cell_spacing1 * cell_spacing2 < 0 or cellerr > 0.01:
                        continue
                     cell_spacing = 0.5 * cell_spacing1 + 0.5 * cell_spacing2

                     # what to do about negative values??
                     if cell_spacing < 15:
                        continue

                     # xalign = hm.hrot(symaxis, 180, xalign @ orig_metal_pos) @ xalign

                     # print(l)
                     # print(l0)
                     # print(n)
                     # print(p0)
                     # print('isect', isect)
                     # assert 0

                     #

                     symaxis = xalign @ symaxis
                     symaxisd = xalign @ symaxisd
                     symaxisd2 = xalign @ symaxisd2

                     # print(np.degrees(target_angle))
                     # print(np.degrees(hm.line_angle(symaxis, pept_axis)))
                     # check for 2nd D2 axis
                     ok1 = 0.03 > hm.line_angle(symaxisd, xspec.axis2d)
                     ok2 = 0.03 > hm.line_angle(symaxisd2, xspec.axis2d)

                     if not (ok1 or ok2):
                        continue
                     if ok2:
                        symaxisd, symaxisd2 = symaxisd2, symaxisd
                     if np.pi / 2 < hm.angle(symaxisd, xspec.axis2d):
                        symaxisd = -symaxisd

                     xyzmetal = xalign @ orig_metal_pos
                     xd2 = [
                        np.array(hm.hrot(symaxis, 180, xyzmetal) @ xalign),
                        np.array(hm.hrot(symaxisd, 180, xyzmetal) @ xalign),
                        np.array(hm.hrot(symaxisd2, 180, xyzmetal) @ xalign),
                     ]
                     if np.any(rpxbody.intersect(rpxbody, xalign, xd2, mindis=kw.clash_dis)):
                        # print('d2 clash')
                        continue

                     frames = rp.geom.expand_xforms_rand(
                        [
                           hm.hrot(xspec.axis1, 120, xspec.orig1 * cell_spacing),
                           hm.hrot(xspec.axis1, 240, xspec.orig1 * cell_spacing),
                           np.array(hm.hrot(symaxis, 180, xyzmetal) @ xalign),
                           np.array(hm.hrot(symaxisd, 180, xyzmetal) @ xalign),
                           np.array(hm.hrot(symaxisd2, 180, xyzmetal) @ xalign),
                        ],
                        depth=10,
                        radius=30.0,
                        cen=xyzmetal[:3],
                     )

                     print(
                        f'{count:5} {iaa:3} {ires:3} {irot:5} {ibonddof:2} ang axisd',
                        f'{hm.angle_degrees(symaxisd, xspec.axis2d):7.3}',
                        f'{hm.angle_degrees(symaxisd2, xspec.axis2d):7.3}',
                        f'{cell_spacing: 7.3}',
                     )

                     outpose0 = mof.util.mutate_one_res(
                        pose,
                        ires + 1,
                        aa,
                        rotcloud.rotchi[irot],
                        sym_num,
                     )

                     def addZN(pose):
                        # add zn
                        znres = make_residue('ZN')
                        pose.append_residue_by_jump(znres, 1)
                        znresi = len(pose.residues)
                        znpos = xyzVec(*xyzmetal[:3])
                        zndelta = znpos - pose.residue(znresi).xyz(1)
                        for ia in range(1, pose.residue(znresi).natoms() + 1):
                           newxyz = zndelta + pose.residue(znresi).xyz(ia)
                           pose.set_xyz(AtomID(ia, znresi), newxyz)

                     fname = f'test_{count:03}_%i.pdb'
                     for iframe, frame in enumerate([xalign] + xd2):
                        outpose = outpose0.clone()
                        xform_pose(outpose, frame)
                        addZN(outpose)
                        # # set chain letter DOESNT WORK
                        # pi = pyrosetta.rosetta.core.pose.PDBInfo(outpose)
                        # for ir in range(1, outpose.size() + 1):
                        #    print(ir)
                        #    pi.chain(ir, 'ABCD' [iframe])
                        # outpose.pdb_info(pi)

                        outpose.dump_pdb(fname % iframe)

                     assert pose.size() % xspec.nfold1 == 0
                     nres_asym = pose.size() // xspec.nfold1
                     xtal_pose = rosetta.protocols.grafting.return_region(outpose0, 1, nres_asym)

                     xform_pose(xtal_pose, xalign)
                     addZN(xtal_pose)
                     ci = pyrosetta.rosetta.core.io.CrystInfo()
                     ci.A(cell_spacing)  # cell dimensions
                     ci.B(cell_spacing)
                     ci.C(cell_spacing)
                     ci.alpha(90)  # cell angles
                     ci.beta(90)
                     ci.gamma(90)
                     ci.spacegroup(xspec.spacegroup)  # sace group
                     pi = pyrosetta.rosetta.core.pose.PDBInfo(xtal_pose)
                     pi.set_crystinfo(ci)
                     xtal_pose.pdb_info(pi)
                     xtal_pose.dump_pdb('test_xtal.pdb')

                     # os.system(
                     # f'cat test_{count:03}_*.pdb | grep -v HEADER |grep -v LINK > test_{count:03}.pdb '
                     # )
                     # os.system(f'rm test_{count:03}_*.pdb')
                     # assert 0

                     count += 1
                     if count > 0:
                        assert 0

def get_test_kw(kw):
   kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
   print(f'{"":!^80}')
   print(f'{"no pdb list input, using test only_one":!^80}')
   print(f'{str(kw.inputs):!^80}')
   print(f'{"":!^80}')

   kw.spacegroups = ['p23']
   kw.output_prefix = '_mof_main_test_output' + '_'.join(kw.spacegroups) + '/'
   kw.scale_number_of_rotamers = 0.5
   kw.max_bb_redundancy = 0.0  # 0.3
   kw.err_tolerance = 2.0
   kw.dist_err_tolerance = 1.0
   kw.angle_err_tolerance = 15
   kw.min_dist_to_z_axis = 6.0
   kw.sym_axes_angle_tolerance = 6.0
   kw.angle_to_cart_err_ratio = 20.0
   kw.max_dun_score = 4.0
   kw.clash_dis = 3.3
   kw.contact_dis = 7.0
   kw.min_contacts = 0
   kw.max_score_minimized = 50.0
   kw.min_cell_size = 0
   kw.max_cell_size = 50
   kw.max_solv_frac = 0.80
   kw.debug = True
   # kw.continue_from_checkpoints = False
   return kw

_lblmap = dict(
   CYS='lC',
   DCYS='dC',
   ASP='lD',
   DASP='dD',
   GLU='lE',
   DGLU='dE',
   HIS='lH',
   DHIS='dH',
   HISD='lJ',
   DHISD='dJ',
)
