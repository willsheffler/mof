from mof.pyrosetta_init import rosetta as r

# aavol = dict(
#    A=180 * 2.0,
#    D=110 * 2.0,
#    C=110 * 2.0,
#    E=150 * 2.0,
#    G=100 * 2.0,
#    H=150 * 2.0,
#    P=
#    Z=0,
# )

# https://clemlab.sitehost.iu.edu/Publications/pubs/pub%20051.pdf
# residue protein coreb solutionc peptide ionsd extendede
expand_vdw_vol_factor = 1.75
aavol = dict(
   A=expand_vdw_vol_factor * 81.8,
   C=expand_vdw_vol_factor * 110,
   D=expand_vdw_vol_factor * 111.0,
   E=expand_vdw_vol_factor * 131.4,
   F=expand_vdw_vol_factor * 171.7,
   G=expand_vdw_vol_factor * 123.456,  # 56.5,
   H=expand_vdw_vol_factor * 130.7,
   I=expand_vdw_vol_factor * 144.1,
   K=expand_vdw_vol_factor * 131.0,
   L=expand_vdw_vol_factor * 142.8,
   M=expand_vdw_vol_factor * 148.3,
   N=expand_vdw_vol_factor * 115.9,
   P=expand_vdw_vol_factor * 106.4,
   Q=expand_vdw_vol_factor * 134.7,
   R=expand_vdw_vol_factor * 200.12345,
   S=expand_vdw_vol_factor * 89.4,
   T=expand_vdw_vol_factor * 111.5,
   V=expand_vdw_vol_factor * 122.7,
   W=expand_vdw_vol_factor * 210.7,
   Y=expand_vdw_vol_factor * 183.5,
   Z=expand_vdw_vol_factor * 0,
)

def approx_solvent_fraction(pose, xspec, celldim=None):

   # print('celldim', celldim)
   seq = pose.sequence()
   if r.core.pose.symmetry.is_symmetric(pose):
      # print('nasym', nsym)
      nasym = r.core.pose.symmetry.symmetry_info(pose).get_nres_subunit()
      seq = seq[:nasym]

   # assert all(aa in aavol for aa in seq)
   peptvol = r.core.scoring.packing.get_surf_vol(pose, 1.4).tot_vol
   print(f'peptvol {peptvol}')
   vol = xspec.nsubs * peptvol

   # vol = xspec.nsubs * sum(aavol[_] for _ in seq)

   # for ir in range(1, nasym + 1):
   # print(pose.residue(ir).name())
   # vol += xspec.nsubs * aavol[pose.residue(ir).name()]
   celldim = celldim if celldim else pose.pdb_info().crystinfo().A()
   cellvol = celldim**3

   print('approx_solvent_fraction:', vol, celldim, cellvol, pose.size())

   # print(pose.pdb_info().crystinfo().A())
   # print(seq, vol, celldim, cellvol)
   # assert 0
   solvfrac = max(0.0, 1.0 - vol / cellvol)

   return solvfrac
