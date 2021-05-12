import __main__
__main__.pymol_argv = ['pymol', '-qe']

import pymol
pymol.finish_launching()
from pymol import cmd

sys.path.append('/home/sheffler/src/wills_pymol_crap')
import math
from wills_pymol_crap.cgo import cgo_circle
from wills_pymol_crap.pymol_util import mysetview, com, showcyl
from wills_pymol_crap.xyzMath import Vec, alignvector

animation_time = 0.8

def main():
   cmd.load(
      '/home/sheffler/debug/mof/new_1res/new_1res_test_p23/mof_solv1%_c.12.8_0001.pdb.noter.pdb_nres3_cell019_ASP_000014.pdb'
   )
   showcyl(-20 * Vec(1, 1, 1), 20 * Vec(1, 1, 1), 0.1, col=(1, 0, 0))
   showcyl(-20 * Vec(0, 1, 1), 20 * Vec(0, 1, 1), 0.1, col=(0, 0, 1))

   metalpos = Vec(0.00000, 0, 9.795000)
   showcyl(metalpos - 100 * Vec(0, 1, 0), metalpos + 100 * Vec(0, 1, 0), 0.1, col=(0, 1, 0))

   showcyl(Vec(0, 0, 0), Vec(100, 0, 0), 0.1, col=(1, 1, 1))
   showcyl(Vec(0, 0, 0), Vec(0, 100, 0), 0.1, col=(1, 1, 1))
   showcyl(Vec(0, 0, 0), Vec(0, 0, 100), 0.1, col=(1, 1, 1))

def circles():
   cmd.load('znd2.pse')
   cmd.hide('ev', 'axes')
   dof = Vec(1.205000, -1.319000, -1.174000).normalized()

   pept_cen = com('vis and name ca')
   metal_cen = Vec(0, 0, 0)
   showcyl(pept_cen - Vec(20, 20, 20), pept_cen + Vec(20, 20, 20), 0.05, col=(1, 0, 0))
   showcyl(metal_cen + Vec(-34, 0, 0), metal_cen + Vec(34, 0, 0), 0.05, col=(0, 1, 0))
   showcyl(metal_cen - dof * 34, metal_cen, 0.2, col=(0, 0, 1))
   cmd.scene('new', 'append')
   # cmd.hide('ev', 'SEG0')

   showcyl(metal_cen, metal_cen + Vec(20, 20, 20), 0.2, col=(1, 0, 0))
   cmd.scene('new', 'append')

   cgo_circle(metal_cen - dof * 10, r=14.4, col=(0, 1, 0), xform=alignvector(Vec(0, 0, 1), dof),
              w=5)
   cgo_circle(metal_cen + Vec(10, 10, 10) / math.sqrt(3), r=14.4, w=5, col=(1, 0, 0),
              xform=alignvector(Vec(0, 0, 1), Vec(1, 1, 1)))
   cmd.scene('new', 'append')

   showcyl(metal_cen, metal_cen + 20 * Vec(0.03, 0.98, -0.04), 0.1)
   showcyl(metal_cen, metal_cen + 20 * Vec(0.001895 - 0.025, 0.036645 - 0.025, 0.999323), 0.1)

   cmd.zoom()

def xtal_grow():
   cmd.stereo('walleye')
   cmd.set('scene_animation_duration', animation_time)
   # mysetview(Vec(-1, 1, 1))
   # cmd.set('orthoscopic', True)
   cmd.load('test_asym.pdb')
   cmd.remove('hydro')

   cmd.hide('car')
   cmd.color('white', 'test_asym and elem C')
   cmd.symexp('_xtal', 'test_asym', 'name ca', 1000)
   cmd.color('cyan', '_xtal* and elem C')
   cmd.hide('ev', '_xtal*')

   cmd.scene('cryst0', 'store', animate=animation_time)

   for i in range(1, 10):
      cmd.color('green', '_xtal* and elem C and vis')

      if i % 2:
         cmd.show('sti', 'byobj _xtal* within 1.5 of (vis and name N+C)')
      else:
         cmd.show('sti', 'byobj _xtal* within 0.5 of (vis and name FE)')
      cmd.zoom('vis', complete=1)
      cmd.scene('cryst%i' % i, 'store', animate=animation_time)
      n = cmd.select('_xtal* and vis')

      if n == cmd.select('_xtal*'):
         break

   cmd.show('sti', '_xtal*')
   cmd.color('green', '_xtal* and elem C')
   # cmd.set('orthoscopic', False)
   cmd.scene('T0', 'store', animate=animation_time)

   cmd.scene('cryst0')

if __name__ == '__main__':
   main()
