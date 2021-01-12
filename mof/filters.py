from mof.pyrosetta_init import rosetta as r
aavol = dict(
   A=180 * 2.0,
   D=110 * 2.0,
   C=110 * 2.0,
   E=150 * 2.0,
   G=100 * 2.0,
   H=150 * 2.0,
   Z=0,
)

def approx_solvent_fraction(pose, xspec, celldim=None):
   # print('celldim', celldim)
   seq = pose.sequence()
   if r.core.pose.symmetry.is_symmetric(pose):
      # print('nasym', nsym)
      nasym = r.core.pose.symmetry.symm_info(pose).get_nres_subunit()
      seq = seq[:nasym]
   vol = xspec.nsubs * sum(aavol[_] for _ in seq)

   # for ir in range(1, nasym + 1):
   # print(pose.residue(ir).name())
   # vol += xspec.nsubs * aavol[pose.residue(ir).name()]
   celldim = celldim if celldim else pose.pdb_info().crystinfo().A()
   cellvol = celldim**3

   # print(pose.pdb_info().crystinfo().A())
   # print(seq, vol, celldim, cellvol)
   # assert 0
   solvfrac = max(0.0, 1.0 - vol / cellvol)

   return solvfrac
