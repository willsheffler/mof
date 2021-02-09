def xtal_search_single_residue(search_spec, pose, **kw):
   raise NotImplemntedError('xtal_search_single_residue needs updating')
   kw = rp.Bunch(kw)

   spec = search_spec
   xspec = spec.xtal_spec

   results = list()

   p_n = pose.pdb_info().name().split('/')[-1]
   # gets rid of the ".pdb" at the end of the pdb name
   pdb_name = p_n[:-4]

   print(f'{pdb_name} searching')

   # check the symmetry type of the pdb
   last_res = rt.core.pose.chain_end_res(pose).pop()
   total_res = int(last_res)
   sym_num = pose.chain(last_res)
   if sym_num < 2:
      print('bad pdb', p_n)
      return list()
   sym = int(sym_num)
   peptide_sym = "C%i" % sym_num

   for ires in range(1, int(total_res / sym) + 1):
      if pose.residue_type(ires) not in (spec.rts.name_map('GLY'), spec.rts.name_map('ALA'),
                                         spec.rts.name_map('DALA')):
         continue
      lig_poses = util.mut_to_ligand(pose, ires, spec.ligands, spec.sym_of_ligand)
      bad_rots = 0
      for ilig, lig_pose in enumerate(lig_poses):
         mut_res_name, lig_sym = lig_poses[lig_pose]

         rotamers = lig_pose.residue(ires).get_rotamers()
         rotamers = util.extra_rotamers(rotamers, lb=-20, ub=21, bs=20)

         pose_num = 1
         for irot, rotamer in enumerate(rotamers):
            #for i in range(1, len(rotamer)+1): # if I want to sample the metal-axis too
            for i in range(len(rotamer)):
               lig_pose.residue(ires).set_chi(i + 1, rotamer[i])
            rot_pose = rt.protocols.grafting.return_region(lig_pose, 1, lig_pose.size())

            spec.sfxn_rotamer(rot_pose)
            dun_score = rot_pose.energies().residue_total_energy(ires)
            if dun_score >= spec.max_dun_score:
               bad_rots += 1
               continue
            rpxbody = rp.Body(rot_pose)

            ############ fix this

            metal_orig = hm.hpoint(util.coord_find(rot_pose, ires, 'VZN'))
            hz = hm.hpoint(util.coord_find(rot_pose, ires, 'HZ'))
            ne = hm.hpoint(util.coord_find(rot_pose, ires, 'VNE'))
            metal_his_bond = hm.hnormalized(metal_orig - ne)
            metal_sym_axis0 = hm.hnormalized(hz - metal_orig)
            dihedral = xspec.dihedral

            ############# ...

            rots_around_nezn = hm.xform_around_dof_for_vector_target_angle(
               fix=spec.pept_axis, mov=metal_sym_axis0, dof=metal_his_bond,
               target_angle=np.radians(dihedral))

            for idof, rot_around_nezn in enumerate(rots_around_nezn):
               metal_sym_axis = rot_around_nezn @ metal_sym_axis0
               assert np.allclose(hm.line_angle(metal_sym_axis, spec.pept_axis),
                                  np.radians(dihedral))

               newhz = util.coord_find(rot_pose, ires, 'VZN') + 2 * metal_sym_axis[:3]

               aid = rt.core.id.AtomID(rot_pose.residue(ires).atom_index('HZ'), ires)
               xyz = rVec(newhz[0], newhz[1], newhz[2])
               rot_pose.set_xyz(aid, xyz)

               tag = f'{pdb_name}_{ires}_{lig_poses[lig_pose][0]}_{idof}_{pose_num}'
               xtal_poses = mof.xtal_build.xtal_build(
                  pdb_name,
                  xspec,
                  aa1,
                  aa2,
                  rot_pose,
                  peptide_sym,
                  spec.pept_orig,
                  spec.pept_axis,
                  lig_sym,
                  metal_orig,
                  metal_sym_axis,
                  rpxbody,
                  tag,
               )

               if False and xtal_poses:
                  print('hoaktolfhtoia')
                  print(rot_pose)
                  print(pdb_name)
                  print(xspec)
                  rot_pose.dump_pdb('test_xtal_build_p213.pdb')
                  print(ires)
                  print(peptide_sym)
                  print(spec.pept_orig)
                  print(spec.pept_axis)
                  print(lig_sym)
                  print(metal_orig)
                  print(metal_sym_axis)
                  print('rp.Body(pose)')

                  xalign, xpose, bodypdb = xtal_poses[0]

                     mof.result.Result(
                        xspec,
                        fname,
                        xalign,
                        rpxbody,
                        xtal_pose,
                        body_pdb,
                     ))

            pose_num += 1
      # print(pdb_name, 'res', ires, 'bad rots:', bad_rots)

   return results
