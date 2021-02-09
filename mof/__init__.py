from mof import util, xtal_search, rotamer_cloud, xtal_spec, xtal_build, result, options, minimize, filters, app

import numpy, ctypes
if hasattr(numpy.__config__, 'mkl_info'):
   mkl_rt = ctypes.CDLL('libmkl_rt.so')
   mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
