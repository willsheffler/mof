class Result:
   """mof xtal search hit"""
   def __init__(self, xspec, label, xalign, rpxbody, xtal_asym_pose, symbody_pdb):
      super(Result, self).__init__()
      self.xspec = xspec
      self.label = label
      self.xtal_asym_pose = xtal_asym_pose
      self.symbody_pdb = symbody_pdb
      self.xalign = xalign
      self.rpxbody = rpxbody
