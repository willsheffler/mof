import pyrosetta
from mof import data

pyrosetta_flags = f'-mute all -output_virtual -extra_res_fa {data.HZ3_params} {data.HZ4_params} {data.HZD_params} -preserve_crystinfo'

pyrosetta.init(pyrosetta_flags)
