from mof.pyrosetta_init import rosetta as r
aavol = dict(
   A=180 * 1.333,
   D=110 * 1.333,
   C=110 * 1.333,
   E=150 * 1.333,
   H=150 * 1.333,
   Z=0,
)

def approx_solvent_fraction(pose, xspec, celldim=None):
   seq = pose.sequence()
   if r.core.pose.symmetry.is_symmetric(pose):
      nasym = r.core.pose.symmetry.symm_info(pose).get_nres_subunit()
      seq = seq[:nasym]
   vol = xspec.nsubs * sum(aavol[_] for _ in seq)

   # for ir in range(1, nasym + 1):
   # print(pose.residue(ir).name())
   # vol += xspec.nsubs * aavol[pose.residue(ir).name()]
   celldim = celldim if celldim else pose.pdb_info().crystinfo().A()
   cellvol = celldim**3
   return max(0, 1.0 - vol / cellvol)
