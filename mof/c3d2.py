import mof, rpxdock as rp, numpy as np, os, rpxdock.homog as hm, shutil, sys, datetime
from mof.pyrosetta_init import rosetta, xform_pose, make_residue
from mof.pyrosetta_init import (rosetta as r, rts, makelattice, addcst_dis, addcst_ang,
                                addcst_dih, name2aid, printscores, get_sfxn)
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta import AtomID
import pyrosetta

def postprocess_c3d2(kw):
  # print(kw.inputs)
  results = list()
  for fn in kw.inputs:
    print(fn)
    results += rp.load(fn)
  print(len(results))

  assert 0

def is_rosetta_stuff(k, v, d):
  # if not isinstance(v, (int, str, float)):
  # print('  ' * d, k, type(v))
  if isinstance(v, r.core.pose.Pose):
    # print('   ' * d, 'found a pose', k)
    return True

def strip_rosetta_content_from_results(kw):
  for fn in kw.inputs:
    r = rp.load(fn)
    r.visit_remove_if(is_rosetta_stuff)
    newfn = os.path.dirname(fn) + '/noposes_' + os.path.basename(fn)
    print('dumping', newfn)
    rp.dump(r, newfn)

def align_cx_pose_to_z(pose, fname):
  pose.dump_pdb(f'original.pdb')
  b = rp.Body(pose)
  # print(f'pose.size() {pose.size():7.3f}')
  nasym = pose.size() // 3
  x = (b.stub[nasym + 1]) @ np.linalg.inv(b.stub[1])
  axis, ang = rp.homog.axis_angle_of(x)
  if not np.allclose(ang, np.pi * 2 / 3, atol=0.04):
    print("WARNING input not C3?", np.degrees(ang), fname)
    return False
  x = rp.homog.align_vector(axis, [0, 0, 1, 0])
  x[:, 3] = x @ -b.com()
  xform_pose(pose, x)
  # print(axis, np.degrees(ang))
  pose.dump_pdb(f'aligned.pdb')

  return pose

def main_loop_c3d2():

  # for i in range(3):
  # print('sbtdnbawiethli3q4jgyluahieorsvbnrasdsenh')
  # os.system('rm test_*.pdb')

  kw = mof.options.get_cli_args()
  if kw.postprocess: return postprocess_c3d2(kw)
  if kw.strip_rosetta_content_from_results: return strip_rosetta_content_from_results(kw)

  kw.timer = rp.Timer().start()
  if kw.test_run:
    kw = get_test_kw(kw)
  print_options(kw)

  if kw.overwrite and os.path.exists(os.path.dirname(kw.output_prefix)):
    print(kw.output_prefix)
    shutil.rmtree(os.path.dirname(kw.output_prefix))
  assert kw.output_prefix

  outdir = os.path.dirname(kw.output_prefix)
  if not outdir:
    outdir = kw.output_prefix + '/'
    # kw.output_prefix = kw.output_prefix + '\n' # this was a nice one!
    kw.output_prefix = kw.output_prefix + '/' + kw.output_prefix

  print(f'os.makedirs({outdir + "clustered/"}, exist_ok=True)')
  os.makedirs(outdir + 'clustered/', exist_ok=True)

  if not kw.spacegroups:
    print('NO spacegroups specified (e.g. --spacegroups P23)')
    return
  # for k, v in kw.items():
  #    try:
  #       print(k, v)
  #    except ValueError:
  #       print(k, type(v))
  # kw.aa_labels = 'ASP DASP CYS DCYS HIS DHIS HISD DHISD GLU DGLU'.split()
  # kw.aa_labels = 'ASP DASP CYS DCYS HIS DHIS GLU DGLU'.split()
  if 'HISD' in kw.aa_labels:
    print('removing HISD, crashes for some reason')
    kw.aa_labels.remove('HISD')
  if 'DHISD' in kw.aa_labels:
    print('removing DHISD, crashes for some reason')
    kw.aa_labels.remove('DHISD')
  rotclouds = mof.rotamer_cloud.get_rotclouds(**kw)

  pept_axis = np.array([0, 0, 1, 0])
  tetrahedral_angle = 109.47122063449069
  stepdegrees = 3

  results = list()
  rfname = f'{kw.output_prefix}_results.pickle'
  print('fname prefix:', kw.output_prefix)
  print('result fname:', rfname)

  for ipdbpath, pdbpath in enumerate(kw.inputs):

    pose = rosetta.core.import_pose.pose_from_file(pdbpath)
    if not align_cx_pose_to_z(pose, pdbpath):
      continue

    rpxbody = rp.Body(pose)

    print()
    print(f'{" %i of %i "%(ipdbpath+1 , len(kw.inputs)):#^80}')
    print(pdbpath)
    print(f'{"":#^80}')

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

      # sym_num = pose.chain(len(pose.residues))
      sym_num = xspec.nfold1
      nresasym = len(pose.residues) // sym_num
      # print(sym_num, nresasym)
      # assert 0

      for iaa, aa in enumerate(kw.aa_labels):

        rotcloud = rotclouds[_lblmap[aa]]

        for ires in range(nresasym):

          print(f'{f" LOOP {spacegroup} {aa} {ires} ":*^80}')
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
                if d != 0: cell_spacing1 = n / d
              for n, d in zip(axis2_orig[:3], xspec.orig2[:3]):
                if d != 0: cell_spacing2 = n / d
              cellerr = (abs(cell_spacing1 - cell_spacing2) /
                         (abs(cell_spacing1) + abs(cell_spacing2)))

              if cell_spacing1 * cell_spacing2 < 0 or cellerr > 0.04:
                continue
              # cell_spacing = 0.5 * cell_spacing1 + 0.5 * cell_spacing2
              cell_spacing = cell_spacing2  # keep d2 center intact
              # print('cell_spacing', cell_spacing1, cell_spacing2, cellerr)

              # what to do about negative values??
              cell_spacing = abs(cell_spacing)
              if abs(cell_spacing) < 10:
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
              frames1 = [
                 hm.hrot(xspec.axis1, 120, xspec.orig1 * cell_spacing),
                 hm.hrot(xspec.axis1, 240, xspec.orig1 * cell_spacing),
                 np.array(hm.hrot(symaxis, 180, xyzmetal) @ xalign),
                 np.array(hm.hrot(symaxisd, 180, xyzmetal) @ xalign),
                 np.array(hm.hrot(symaxisd2, 180, xyzmetal) @ xalign),
              ]
              if np.any(rpxbody.intersect(rpxbody, xalign, frames1, mindis=kw.clash_dis)):
                # print('d2 clash')
                continue

              # kw.timer.checkpoint('before expand_xforms_rand')
              # frames, meta = rp.geom.expand_xforms_rand(
              #    [
              #       hm.hrot(xspec.axis1, 120, xspec.orig1 * cell_spacing),
              #       hm.hrot(xspec.axis1, 240, xspec.orig1 * cell_spacing),
              #       np.array(hm.hrot(symaxis, 180, xyzmetal) @ xalign),
              #       np.array(hm.hrot(symaxisd, 180, xyzmetal) @ xalign),
              #       np.array(hm.hrot(symaxisd2, 180, xyzmetal) @ xalign),
              #    ],
              #    depth=10,
              #    radius=30.0,
              #    cen=xyzmetal[:3],
              # )
              # clash = False
              # for iframe, frame in enumerate(frames):
              #    # if np.any(rpxbody.intersect(rpxbody, xalign, frame, mindis=kw.clash_dis)):
              #    #    if not np.any(rpxbody.intersect(rpxbody, xalign, frame, mindis=0.01)):
              #    #       clash = True
              #    #       break
              #    p = pose.clone()
              #    xform_pose(p, frame)
              #    p.dump_pdb('woo%i.pdb' % iframe)
              # assert 0
              # if clash: continue
              # kw.timer.checkpoint('after expand_xforms_rand')
              # kw.timer.report()
              # assert 0

              outpose0 = mof.util.mutate_one_res(
                 pose,
                 ires + 1,
                 aa,
                 rotcloud.rotchi[irot],
                 sym_num,
              )

              #
              def addZN(pose):
                # add zn
                znres = make_residue('VZN')
                pose.append_residue_by_jump(znres, 1)
                znresi = len(pose.residues)
                znpos = xyzVec(*xyzmetal[:3])
                zndelta = znpos - pose.residue(znresi).xyz(1)
                for ia in range(1, pose.residue(znresi).natoms() + 1):
                  newxyz = zndelta + pose.residue(znresi).xyz(ia)
                  pose.set_xyz(AtomID(ia, znresi), newxyz)

              # fname = f'test_{count:03}_%i.pdb'
              # for iframe, frame in enumerate([xalign] + xd2):
              #    outpose = outpose0.clone()
              #    xform_pose(outpose, frame)
              #    addZN(outpose)
              #    # # set chain letter DOESNT WORK
              #    # pi = pyrosetta.rosetta.core.pose.PDBInfo(outpose)
              #    # for ir in range(1, outpose.size() + 1):
              #    #    print(ir)
              #    #    pi.chain(ir, 'ABCD' [iframe])
              #    # outpose.pdb_info(pi)
              #    outpose.dump_pdb(fname % iframe)

              assert pose.size() % xspec.nfold1 == 0
              nres_asym = pose.size() // xspec.nfold1
              xtal_pose = rosetta.protocols.grafting.return_region(outpose0, 1, nres_asym)
              set_cell_params(xtal_pose, xspec, cell_spacing)

              #
              pdb_name = os.path.basename(pdbpath)

              lattice_pose = xtal_pose.clone()
              makelattice(lattice_pose)
              syminfo = rosetta.core.pose.symmetry.symmetry_info(lattice_pose)
              surfvol = rosetta.core.scoring.packing.get_surf_vol(lattice_pose, 1.4)
              peptvol = 0.0
              for ir, r in enumerate(lattice_pose.residues):
                ir = ir + 1  # rosetta numbering
                for ia in range(1, r.natoms() + 1):
                  v = surfvol.vol[AtomID(ia, ir)]
                  if not np.isnan(v): peptvol += v
              peptvol /= syminfo.subunits()
              peptvol *= xspec.nsubs
              print(f'peptvol {peptvol}')

              # solv_frac = mof.filters.approx_solvent_fraction(xtal_pose, xspec, cell_spacing)
              solv_frac = max(0.0, 1.0 - peptvol / cell_spacing**3)
              if kw.max_solv_frac < solv_frac:
                print('     ', xspec.spacegroup, pdb_name, aa, 'Fail on solv_frac', solv_frac)
                continue
              else:
                print('     ', xspec.spacegroup, pdb_name, aa, 'Win  on solv_frac', solv_frac)

              #

              xform_pose(xtal_pose, xalign)
              addZN(xtal_pose)

              sfxn_min = get_sfxn('minimize')
              xtal_pose_min, mininfo = minimize_oneres(
                 sfxn_min,
                 xspec,
                 xtal_pose,
                 **kw,
              )
              if xtal_pose_min is None:
                print(f'min failed, xtal_pose_min is None')
                continue

              print(f'{len(results):5} {iaa:3} {ires:3} {irot:5} {ibonddof:2} ang axisd',
                    f'{hm.angle_degrees(symaxisd, xspec.axis2d):7.3}',
                    f'{hm.angle_degrees(symaxisd2, xspec.axis2d):7.3}', f'{cell_spacing:7.3}',
                    f'farep {mininfo.score_fa_rep:7.3}', f'solv {solv_frac:5.3}')

              # for st, score in _bestscores.items():
              # if score > 1.0: print('best', st, score)

              if mininfo.score_fa_rep > kw.max_score_minimized / 3 * nresasym:
                # if mininfo.score_wo_cst > kw.max_score_minimized:
                print(f'score_fa_rep fail {mininfo.score_fa_rep}')
                continue

              tag = ''.join([
                 f'_solv{int(solv_frac*100):02}_',
                 f'{os.path.basename(pdbpath)}',
                 f'_nres{nresasym}',
                 f'_cell{int(cell_spacing):03}_',
                 f'{aa}_',
                 f'{len(results):06}',
              ])
              fn = kw.output_prefix + tag + '.pdb'
              # print('dump intermediate', fn)
              # assert 0

              result = prepare_result(**vars(), **kw)
              result.pdb_fname = fn
              result.spacegroup = spacegroup
              result.aa = aa
              result.seq = xtal_pose_min.sequence()
              result.ires = ires
              result.irot = irot
              result.ibonddof = ibonddof
              results.append(result)

              print(f'{f" DUMP PDB {fn} ":!^100}')
              rp.dump(result, fn[:-4] + '.pickle')
              xtal_pose_min.dump_pdb(fn)

  # if os.path.exists(rfname):
  #   # print('type(results)', type(results))
  #   # prev_results = rp.load(rfname)
  #   # print('nprev', len(prev_results), 'n', len(results))
  #   # new_results = prev_results + results
  #   # results = new_results

  #   print('READING PREV RESULTS', datetime.datetime.now(), 'Ncur', len(results))
  #   results = rp.load(rfname) + results

  # rp.dump(results, rfname)
  # print('SAVED', len(results), 'RESULTS', datetime.datetime.now())

  if not results:
    print(f'{"":!^100}')
    print('NO RESULTS!!!')
    print(f'{"":!^100}')
    print('DONE')

  return

def cluster_stuff():
  # cluster
  scores = np.array([r.info.score_fa_rep for r in results])
  sorder = np.argsort(scores)  # perm to sorted

  clustcen = np.arange(len(scores), dtype=np.int)
  if kw.cluster:
    print(f'{" CLUSTER ":#^100}')
    # crds sorted by score
    crd = np.stack([r.info.bbcoords for r in results])
    crd = crd[sorder].reshape(len(scores), -1)
    nbbatm = (results[0].asym_pose_min.size() - 1) * 3
    clustcen = rp.cluster.cookie_cutter(crd, kw.max_bb_redundancy * np.sqrt(nbbatm))
    clustcen = sorder[clustcen]  # back to original order
    results = [results[i] for i in clustcen]
    kw.timer.checkpoint('filter_redundancy')

    for iclust, r in enumerate(results):
      outdir = kw.output_prefix + os.path.dirname(r.info.tag)
      if not outdir: outdir = '.'
      clust_fname = outdir + '/clustered/clust%06i_' % iclust + os.path.basename(
         r.info.tag) + '.pdb'
      # print("CLUST DUMP", r.info.tag, clust_fname)
      r.asym_pose_min.dump_pdb(clust_fname)

  return

  # dump pdbs
  #   for i, result in enumerate(results):
  #      fname = kw.output_prefix + 'asym_' + result.info.label + '.pdb'
  #      print('dumping', fname)
  #      result.info.fname = fname
  #      result.asym_pose_min.dump_pdb(fname)
  #      # rp.util.dump_str(result.symbody_pdb, 'sym_' + result.info.label + '.pdb')
  # kw.timer.checkpoint('dump_pdbs')

  # make Dataset
  # results = mof.result.results_to_xarray(results)
  # results['clustcen'] = ('result', clustcen)
  # results.attrs['kw'] = kw

  # print('saving results to', rfname)
  # rp.dump(results, rfname)
  # print(f'{" RESULTS ":=^80}')
  # print(results)
  # print(f'{" END RESULTS ":=^80}')
  # kw.timer.checkpoint('dump_info')

def prepare_result(
   xtal_pose_min,
   xspec,
   tag,
   xalign,
   pdb_name,
   mininfo,
   solv_frac,
   iaa,
   ncontact=0,
   enonbonded=0,
   **kw,
):
  kw = rp.Bunch(kw)
  celldim = xtal_pose_min.pdb_info().crystinfo().A()
  label = f"{pdb_name}_{xspec.spacegroup.replace(' ','_')}_{tag}_cell{int(celldim):03}_ncontact{ncontact:02}_score{int(enonbonded):03}"

  info = mininfo.sub(  # adding to mininfo
     label=label,
     xalign=xalign,
     # ncontact=ncontact,
     # enonbonded=enonbonded,
     sequence=','.join(r.name() for r in xtal_pose_min.residues),
     solv_frac=solv_frac,
     celldim=celldim,
     spacegroup=xspec.spacegroup,
     nsubunits=xspec.nsubs,
     nres=xtal_pose_min.size() - 1,
     tag=tag,
  )
  bbcoords = np.array([(v[0], v[1], v[2]) for v in [[r.xyz(n)
                                                     for n in ('N', 'CA', 'C')]
                                                    for r in xtal_pose_min.residues[:-1]]])
  # allow for different num of res
  bbpad = 9e9 * np.ones(shape=(kw.max_pept_size - xtal_pose_min.size() + 1, 3, 3))
  # separate aas in hacky way
  bbpad[-1] = 10000 * iaa
  info['bbcoords'] = np.concatenate([bbcoords, bbpad])

  return rp.Bunch(
     xspec=xspec,
     # rpxbody=rpxbody,
     # asym_pose=xtal_pose,
     asym_pose_min=xtal_pose_min,
     info=info,
  )

def minimize_oneres(sfxn, xspec, pose, debug=False, **kw):
  debug = False

  kw = rp.Bunch(kw)

  nresasym = pose.size()
  beg = 1
  end = nresasym - 1
  metalres = rts.name_map('ZN')
  metalname = 'ZN'
  metalresno = nresasym
  metalnbonds = 4

  metalaid = AtomID(1, metalresno)

  # cst_ang_metal = 109.47
  # cst_dis_metal = 2.2
  # cst_sd_metal_olap = 0.01
  # cst_sd_metal_dir = 0.4
  # cst_sd_metal_lig_dist = 0.2
  # cst_sd_metal_lig_ang = 0.4
  # cst_sd_metal_coo = 0.5
  # cst_sd_cut_dis = 0.01
  # cst_sd_cut_ang = 0.01
  # cst_sd_cut_dih = 0.1

  pose = pose.clone()
  r.core.pose.remove_lower_terminus_type_from_pose_residue(pose, beg)
  r.core.pose.remove_upper_terminus_type_from_pose_residue(pose, end)
  # WTF is this about?
  # for ir in range(1, pose.size() + 1):
  #    if 'HIS' in pose.residue(ir).name():
  #       newname = pose.residue(ir).name().replace('HIS', 'HIS_D')
  #       newname = newname.replace('_D_D', '')
  #       r.core.pose.replace_pose_residue_copying_existing_coordinates(
  #          pose, ir, rts.name_map(newname))

  if False:
    tmp = pose.clone()
    r.core.pose.replace_pose_residue_copying_existing_coordinates(tmp, metalresno, metalres)
    makelattice(tmp)
    tmp.dump_pdb('before.pdb')
  makelattice(pose)
  if debug: print(f'minimize.py score initial.................... {sfxn(pose):10.3f}')
  # print_nonzero_energies(sfxn, pose)

  syminfo = r.core.pose.symmetry.symmetry_info(pose)
  symdofs = syminfo.get_dofs()
  allowed_jumps = list()
  # for jid, dofinfo in symdofs.items():
  #    # if dofinfo.allow_dof(1) and not any(dofinfo.allow_dof(i) for i in range(2, 7)):
  #    allowed_jumps.append(jid)
  # print('minimize_mof_xtal allowed jumps:', allowed_jumps)
  nxyz = pose.residue(beg).xyz('N')
  cxyz = pose.residue(end).xyz('C')
  nac, cac = None, None  # (N/C)-(a)djacent (c)hain
  for isub in range(1, syminfo.subunits()):
    othern = pose.residue((isub + 0) * nresasym + 1).xyz('N')
    otherc = pose.residue((isub + 1) * nresasym - 1).xyz('C')
    if nxyz.distance(otherc) < kw.bb_break_dist: cac = (isub + 1) * nresasym - 1
    if cxyz.distance(othern) < kw.bb_break_dist: nac = (isub + 0) * nresasym + 1
  if not (nac and cac):
    print('backbone is weird? probably subunits too broken')
    print(syminfo.subunits())
    print('nac   ', nac)
    print('othern', othern)
    print('cac   ', cac)
    print('otherc', otherc)
    # assert 0

    pose.dump_pdb(f'backbone_is_weird.pdb')
    assert 0
    return None, None
  else:
    print(f'backbone is not weird')
  if debug: print('peptide connection 1:', cac, beg)
  if debug: print('peptide_connection 2:', end, nac)
  # pose.dump_pdb('check_cuts.pdb')
  # assert 0

  f_metal_lig_dist = r.core.scoring.func.HarmonicFunc(kw.cst_dis_metal, kw.cst_sd_metal_lig_dist)
  f_metal_lig_ang = r.core.scoring.func.HarmonicFunc(np.radians(kw.cst_ang_metal),
                                                     kw.cst_sd_metal_lig_ang)
  f_metal_olap = r.core.scoring.func.HarmonicFunc(0.0, kw.cst_sd_metal_olap)
  f_point_at_metal = r.core.scoring.func.HarmonicFunc(0.0, kw.cst_sd_metal_dir)
  f_metal_coo = r.core.scoring.func.CircularHarmonicFunc(0.0, kw.cst_sd_metal_coo)
  f_cut_dis = r.core.scoring.func.HarmonicFunc(1.328685, kw.cst_sd_cut_dis)
  f_cut_ang_cacn = r.core.scoring.func.HarmonicFunc(2.028, kw.cst_sd_cut_ang)
  f_cut_ang_cnca = r.core.scoring.func.HarmonicFunc(2.124, kw.cst_sd_cut_ang)
  f_cut_dih = r.core.scoring.func.CircularHarmonicFunc(np.pi, kw.cst_sd_cut_dih)
  f_cut_dihO = r.core.scoring.func.CircularHarmonicFunc(0.00, kw.cst_sd_cut_dih)

  ################### check cutpoint ##################

  conf = pose.conformation().clone()
  assert r.core.conformation.symmetry.is_symmetric(conf)
  # print(pose.pdb_info().crystinfo())
  pi = pose.pdb_info()
  # conf.detect_bonds()
  conf.declare_chemical_bond(cac, 'C', beg, 'N')
  # for iframe, f
  # conf.declare_chemical_bond(end, 'N', nac, 'N')
  pose.set_new_conformation(conf)
  pose.set_new_energies_object(r.core.scoring.symmetry.SymmetricEnergies())
  pose.pdb_info(pi)
  if debug: print(f'minimize.py: score after chem bonds.......... {sfxn(pose):10.3f}')

  #############################################3

  cst_cut, cst_lig_dis, cst_lig_ang, cst_lig_ori = list(), list(), list(), list()

  ############### chainbreaks ################3

  # this doesn't behave well...
  # # 39 C / 1 OVU1
  # # 39 OVL1 / 1 N
  # # 29 OVL2 / CA
  # r.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_UPPER', beg)
  # r.core.pose.add_variant_type_to_pose_residue(pose, 'CUTPOINT_LOWER', end)
  # cres1 = pose.residue(cac)
  # nres1 = pose.residue(1)
  # cres2 = pose.residue(end)
  # nres2 = pose.residue(nac)
  # # Apc = r.core.scoring.constraints.AtomPairConstraint
  # # getaid = lambda p, n, i: AtomID(p.residue(i).atom_index(n.strip()), i)
  # for cst in [
  #       Apc(getaid(pose, 'C   ', cac), getaid(pose, 'OVU1', beg), f_cut),
  #       Apc(getaid(pose, 'OVL1', cac), getaid(pose, 'N   ', beg), f_cut),
  #       Apc(getaid(pose, 'OVL2', cac), getaid(pose, 'CA  ', beg), f_cut),
  #       Apc(getaid(pose, 'C   ', end), getaid(pose, 'OVU1', nac), f_cut),
  #       Apc(getaid(pose, 'OVL1', end), getaid(pose, 'N   ', nac), f_cut),
  #       Apc(getaid(pose, 'OVL2', end), getaid(pose, 'CA  ', nac), f_cut),
  # ]:
  #    pose.add_constraint(cst)

  cst_cut.append(addcst_dis(pose, cac, 'C ', beg, 'N', f_cut_dis))
  cst_cut.append(addcst_dis(pose, end, 'C ', nac, 'N', f_cut_dis))
  if debug: print(f'minimize.py: score after chainbreak dis...... {sfxn(pose):10.3f}')
  cst_cut.append(addcst_ang(pose, cac, 'CA', cac, 'C', beg, 'N ', f_cut_ang_cacn))
  cst_cut.append(addcst_ang(pose, cac, 'C ', beg, 'N', beg, 'CA', f_cut_ang_cnca))
  cst_cut.append(addcst_ang(pose, end, 'CA', end, 'C', nac, 'N ', f_cut_ang_cacn))
  cst_cut.append(addcst_ang(pose, end, 'C ', nac, 'N', nac, 'CA', f_cut_ang_cnca))
  if debug: print(f'minimize.py: score after chainbreak ang...... {sfxn(pose):10.3f}')
  # print(r.numeric.dihedral(
  #       pose.residue(cac).xyz('CA'),
  #       pose.residue(cac).xyz('C'),
  #       pose.residue(beg).xyz('N'),
  #       pose.residue(beg).xyz('CA'),
  #    ))
  cst_cut.append(addcst_dih(pose, cac, 'CA', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dih))
  cst_cut.append(addcst_dih(pose, end, 'CA', end, 'C', nac, 'N ', nac, 'CA', f_cut_dih))
  cst_cut.append(addcst_dih(pose, cac, 'O ', cac, 'C', beg, 'N ', beg, 'CA', f_cut_dihO))
  cst_cut.append(addcst_dih(pose, end, 'O ', end, 'C', nac, 'N ', nac, 'CA', f_cut_dihO))
  if debug: print(f'minimize.py: score after chainbreak dihedral. {sfxn(pose):10.3f}')

  ############## metal constraints ################

  xyzorig = pose.residue(nresasym).xyz(metalname)
  # xyznew = xyzVec(xyzorig)
  znzncount = 0
  for ir in range(2 * nresasym, len(pose.residues), nresasym):
    if pose.residue(ir).is_virtual_residue(): break
    xyz = pose.residue(ir).xyz(metalname)
    if (xyzorig - xyz).length() < 3.0:
      # print('CONTACT ASYM ZN')
      # xyznew += xyz
      addcst_dis(pose, metalresno, metalname, ir, metalname, f_metal_olap)
      znzncount += 1
  assert znzncount is 3
  # xyznew[0] = xyznew[0] / 4
  # xyznew[1] = xyznew[1] / 4
  # xyznew[2] = xyznew[2] / 4
  # pose.set_xyz(AtomID(pose.residue(nresasym).atom_index(metalname), nresasym), xyznew)
  # pose.dump_pdb('foo.pdb')
  # assert 0

  if debug:
    print(f'minimize.py: score after metal olap ......... {sfxn(pose):10.3f}')

  allowed_elems = 'NOS'
  znpos = pose.residue(metalresno).xyz(1)
  znbonded = list()
  for ir in range(1, len(pose.residues) + 1):
    res = pose.residue(ir)
    if not res.is_protein(): continue
    for ia in range(5, res.nheavyatoms() + 1):
      aid = AtomID(ia, ir)
      elem = res.atom_name(ia).strip()[0]
      if elem in allowed_elems:
        xyz = pose.xyz(aid)
        dist = xyz.distance(znpos)
        if dist < 3.5:
          if res.atom_name(ia) in (' OD2', ' OE2'):  # other COO O sometimes closeish
            continue
          znbonded.append(aid)
  if len(znbonded) != metalnbonds:
    print('WRONG NO OF LIGANDING ATOMS', len(znbonded))
    if debug:
      for aid in znbonded:

        print(pose.residue(aid.rsd()).name(), pose.residue(aid.rsd()).atom_name(aid.atomno()))
        pose.dump_pdb('WRONG_NO_OF_LIGANDING_ATOMS.pdb')
        return None, None
      # raise ValueError(f'WRONG NO OF LIGANDING ATOMS {len(znbonded)}')

  # metal/lig distance constraints

  for i, aid in enumerate(znbonded):
    cst = r.core.scoring.constraints.AtomPairConstraint(metalaid, aid, f_metal_lig_dist)
    cst_lig_dis.append(cst)
    pose.add_constraint(cst)

  if debug:
    print(f'minimize.py: score after metal dist ......... {sfxn(pose):10.3f}')
    for aid in znbonded:
      print(aid.rsd(), aid.atomno(),
            pose.residue(aid.rsd()).name(),
            pose.residue(aid.rsd()).atom_name(aid.atomno()))
  # assert 0

  # lig/metal/lig angle constraints (or dihedral in-place constraint for COO)
  # TODO kinda hacky... will need to be more general?

  for i, aid in enumerate(znbonded):
    ir, res = aid.rsd(), pose.residue(aid.rsd())
    if all(_ not in res.name() for _ in 'ASP CYS HIS GLU'.split()):
      return None, None
      assert 0, f'unrecognized res {res.name()}'
    if any(_ in res.name() for _ in 'ASP GLU'.split()):
      # metal comes off of OD1/OE1
      ir, coo = aid.rsd(), ('OD1 CG OD2' if 'ASP' in res.name() else 'OE1 CD OE2').split()
      cst_lig_ori.append(
         addcst_dih(pose, ir, coo[0], ir, coo[1], ir, coo[2], metalaid.rsd(), metalname,
                    f_metal_coo))
    else:
      if 'HIS' in res.name(): aname = 'HD1' if res.has('HD1') else 'HE2'
      if 'CYS' in res.name(): aname = 'HG'
      cst_lig_ori.append(
         addcst_ang(pose, ir, res.atom_name(aid.atomno()), metalaid.rsd(), metalname, ir, aname,
                    f_point_at_metal))

  if debug: print(f'minimize.py: score after metal dir........... {sfxn(pose):10.3f}')

  # for i, iaid in enumerate(znbonded):
  #    for j, jaid in enumerate(znbonded[:i]):
  #       # pripnt(i, j)
  #       cst = r.core.scoring.constraints.AngleConstraint(iaid, metalaid, jaid, f_metal_lig_ang)
  #       cst_lig_ang.append(cst)
  #       pose.add_constraint(cst)

  if debug: print(f'minimize.py: score after lig angle added..... {sfxn(pose):10.3f}')

  ################ minimization #########################

  movemap = r.core.kinematics.MoveMap()
  movemap.set_bb(True)
  movemap.set_chi(True)
  movemap.set_jump(False)
  # for i in allowed_jumps:
  # movemap.set_jump(True, i)
  minimizer = r.protocols.minimization_packing.symmetry.SymMinMover(
     movemap, sfxn, 'lbfgs_armijo_nonmonotone', 0.001, True)  # tol, nblist
  # if sfxn.has_nonzero_weight(r.core.scoring.ScoreType.cart_bonded):
  # minimizer.cartesian(True)
  minimizer.apply(pose)
  if debug:
    print(f'minimize.py: score after min no scale........ {sfxn(pose):10.3f}')
    pose.remove_constraints()
    print(f'minimize.py: score (no cst) after min no scale........ {sfxn(pose):10.3f}')

  # printscores(sfxn, pose)

  kw.timer.checkpoint(f'min scale 1.0')

  asym = r.core.pose.Pose()
  r.core.pose.symmetry.extract_asymmetric_unit(pose, asym, False)
  r.core.pose.replace_pose_residue_copying_existing_coordinates(asym, metalresno, metalres)
  # asym.dump_pdb('asym.pdb')

  # for ir in metalresnos:
  #    r.core.pose.replace_pose_residue_copying_existing_coordinates(pose, ir, metalres)
  # pose.dump_pdb('zafter.pdb')

  if debug: print(kw.timer)

  info = rp.Bunch()
  info.score = sfxn(pose)

  ############### score component stuff ################
  st = r.core.scoring.ScoreType
  etot = pose.energies().total_energies()
  info.score_fa_atr = (etot[st.fa_atr])
  info.score_fa_rep = (etot[st.fa_rep])
  info.score_fa_sol = (etot[st.fa_sol])
  info.score_lk_ball = (etot[st.lk_ball] + etot[st.lk_ball_iso] + etot[st.lk_ball_bridge] +
                        etot[st.lk_ball_bridge_uncpl])
  info.score_fa_elec = (etot[st.fa_elec] + etot[st.fa_intra_elec])
  info.score_hbond_sr_bb = (etot[st.hbond_sr_bb] + etot[st.hbond_lr_bb] + etot[st.hbond_bb_sc] +
                            etot[st.hbond_sc])
  info.score_dslf_fa13 = (etot[st.dslf_fa13])
  info.score_atom_pair_constraint = (etot[st.atom_pair_constraint])
  info.score_angle_constraint = (etot[st.angle_constraint])
  info.score_dihedral_constraint = (etot[st.dihedral_constraint])
  info.score_omega = (etot[st.omega])
  info.score_rotamer = (etot[st.fa_dun] + etot[st.fa_dun_dev] + etot[st.fa_dun_rot] +
                        etot[st.fa_dun_semi] + etot[st.fa_intra_elec] + etot[st.fa_intra_rep] +
                        etot[st.fa_intra_atr_xover4] + etot[st.fa_intra_rep_xover4] +
                        etot[st.fa_intra_sol_xover4])
  info.score_ref = (etot[st.ref])
  info.score_rama_prepro = (etot[st.rama_prepro])
  info.score_cart_bonded = (etot[st.cart_bonded])
  info.score_gen_bonded = (etot[st.gen_bonded])

  ############### cst stuff ################

  pose.remove_constraints()
  info.score_wo_cst = sfxn(pose)
  # record_nonzero_energies(sfxn, pose)

  # for ir in range(1, pose.size() + 1):
  #    rep = pose.energies().residue_total_energies(ir)[st.fa_rep]
  #    if rep != 0: print(ir, 'rep', rep)
  # assert 0

  [pose.add_constraint(cst) for cst in cst_cut]
  info.score_cst_cut = sfxn(pose) - info.score_wo_cst
  pose.remove_constraints()

  [pose.add_constraint(cst) for cst in cst_lig_dis]
  [pose.add_constraint(cst) for cst in cst_lig_ang]
  [pose.add_constraint(cst) for cst in cst_lig_ori]
  info.score_cst_lig_ori = sfxn(pose) - info.score_wo_cst
  pose.remove_constraints()

  [pose.add_constraint(cst) for cst in cst_lig_dis]
  info.score_cst_lig_dis = sfxn(pose) - info.score_wo_cst
  pose.remove_constraints()

  [pose.add_constraint(cst) for cst in cst_lig_ang]
  info.score_cst_lig_ang = sfxn(pose) - info.score_wo_cst
  pose.remove_constraints()

  [pose.add_constraint(cst) for cst in cst_lig_ori]
  info.score_cst_lig_ori = sfxn(pose) - info.score_wo_cst
  pose.remove_constraints()

  return asym, info

def get_test_kw(kw):
  if not kw.inputs:
    kw.inputs = ['mof/data/peptides/c.2.6_0001.pdb']
    print(f'{"":!^80}')
    print(f'{"no pdb list input, using test only_one":!^80}')
    print(f'{str(kw.inputs):!^80}')
    print(f'{"":!^80}')

  kw.spacegroups = ['p23']
  kw.output_prefix = '_mof_main_test_output' + '_'.join(kw.spacegroups) + '/'
  kw.scale_number_of_rotamers = 0.5
  kw.max_bb_redundancy = 2.0
  kw.max_dun_score = 4.0
  kw.clash_dis = 3.3
  kw.contact_dis = 7.0
  kw.min_contacts = 0
  kw.max_score_minimized = 40.0
  kw.min_cell_size = 0
  kw.max_cell_size = 50
  kw.max_solv_frac = 0.80
  kw.debug = True
  # kw.continue_from_checkpoints = False
  return kw

def set_cell_params(xtal_pose, xspec, cell_spacing):
  ci = pyrosetta.rosetta.core.io.CrystInfo()
  ci.A(abs(cell_spacing))  # cell dimensions
  ci.B(abs(cell_spacing))
  ci.C(abs(cell_spacing))
  ci.alpha(90)  # cell angles
  ci.beta(90)
  ci.gamma(90)
  ci.spacegroup(xspec.spacegroup)  # sace group
  pi = pyrosetta.rosetta.core.pose.PDBInfo(xtal_pose)
  pi.set_crystinfo(ci)
  xtal_pose.pdb_info(pi)

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

def print_options(kw):

  print()
  print('!' * 100)
  print()
  print('COMMAND LINE (maybe with whitespace changes):')
  print()
  print(' '.join(sys.argv))
  print()
  print('!' * 100)
  print()

  kw = kw.sub(inputs=None, timer=None)
  longest_key = max(len(k) for k in kw) + 2
  longest_val = max([
     len(repr(v) if isinstance(v, (int, float, str, list)) else str(type(v)))
     for v in kw.values()
  ])
  longest_line = 0
  reprs = list()
  for k, v in kw.items():
    vstr = repr(v) if isinstance(v, (int, float, str, list)) else str(type(v))
    sep1 = '.' * (longest_key - len(k))
    sep2 = ' ' * (longest_val - len(vstr))
    reprs.append((k, sep1, vstr, sep2))
    linelen = len(str(k + sep1 + vstr))
    longest_line = max(longest_line, linelen)

  msg1 = '  THESE OPTION REJECT YOUR REALITY  '
  msg2 = '  AND SUBSTITUTE THEIR OWN  '
  sepline = '  !!         ' + ' ' * (3 + longest_line) + '  !!'
  sepline = sepline + os.linesep + sepline
  print()
  print(f"  {'':!^{longest_line + 18}}")
  print(f"  {msg1:!^{longest_line + 18}}")
  print(f"  {msg2:!^{longest_line + 18}}")
  print(f"  {'':!^{longest_line + 18}}")
  print(sepline)
  for k, sep1, vstr, sep2 in reprs:
    print('  !!       ', k, sep1, vstr, sep2, '  !!')
  print(sepline)
  print(f"  {' END OF OPTIONS ':!^{longest_line + 18}}")
  print()

def print_nonzero_energies(sfxn, pose):
  for st in sfxn.get_nonzero_weighted_scoretypes():
    print(st, pose.energies().total_energies()[st])

_bestscores = dict()

def record_nonzero_energies(sfxn, pose):
  for st in sfxn.get_nonzero_weighted_scoretypes():
    if st not in _bestscores:
      _bestscores[st] = pose.energies().total_energies()[st]
    _bestscores[st] = min(_bestscores[st], pose.energies().total_energies()[st])
