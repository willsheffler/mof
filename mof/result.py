import xarray as xr, numpy as np, rpxdock as rp

# just using a Bunch
# class Result:
#    """mof xtal search hit"""
#    def __init__(self, xspec, rpxbody, xtal_pose, xtal_asym_pose, symbody_pdb, info):
#       super(Result, self).__init__()
#       self.xspec = xspec
#       self.xtal_asym_pose = xtal_asym_pose
#       self.symbody_pdb = symbody_pdb
#       self.rpxbody = rpxbody
#       self.info = info
#
# class Results:
#    def __init__(self, results):
#       self.results = results

def xrdims(k):
   if k == 'xalign': return ['result', 'hrow', 'hcol']

   add bbcoords, use resn# as dimname

   return ['result']

def results_to_xarray(results):
   fields = {k: (xrdims(k), [r.info[k] for r in results]) for k in results[0].info}
   fields['iresult'] = ['result'], np.arange(len(results))
   attrs = {k: [r[k] for r in results] for k in results[0] if k is not 'info'}
   ds = xr.Dataset(fields, attrs=attrs)
   return ds
