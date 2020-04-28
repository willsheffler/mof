import sys, argparse, rpxdock as rp

def default_cli_parser(parent=None, **kw):
   parser = parent if parent else argparse.ArgumentParser(allow_abbrev=False)
   addarg = rp.app.options.add_argument_unless_exists(parser)
   addarg("inputs", nargs="*", type=str, default=[], help='input structures')
   addarg('--max_bb_redundancy', type=float, default=1.0, help='')
   addarg('--err_tolerance', type=float, default=1.5, help='')
   addarg('--dist_err_tolerance', type=float, default=1.0, help='')
   addarg('--angle_err_tolerance', type=float, default=15, help='')
   addarg('--min_dist_to_z_axis', type=float, default=6.0, help='')
   addarg('--sym_axes_angle_tolerance', type=float, default=5.0, help='')
   addarg('--angle_to_cart_err_ratio', type=float, default=20.0, help='')
   addarg('--max_dun_score', type=float, default=4.0, help='')
   addarg('--clash_dis', type=float, default=3.5, help='')
   addarg('--contact_dis', type=float, default=7.0, help='')
   addarg('--min_contacts', type=float, default=30, help='')
   addarg('--max_sym_score', type=float, default=30.0, help='')
   addarg('--min_cell_size', type=float, default=0, help='')
   addarg('--max_cell_size', type=float, default=45, help='')
   addarg('--rotcloud_cache', type=str, default='.rotcloud_cache', help='')
   addarg('--chiresl_his1', type=float, default=3.0, help='')
   addarg('--chiresl_his2', type=float, default=8.0, help='')
   addarg('--chiresl_cys1', type=float, default=6.0, help='')
   addarg('--chiresl_cys2', type=float, default=8.0, help='')
   addarg('--chiresl_asp1', type=float, default=8.0, help='')
   addarg('--chiresl_asp2', type=float, default=5.0, help='')
   addarg('--chiresl_glu1', type=float, default=6.0, help='')
   addarg('--chiresl_glu2', type=float, default=12.0, help='')
   addarg('--chiresl_glu3', type=float, default=6.0, help='')
   addarg('--maxdun_cys', type=float, default=4.0, help='')
   addarg('--maxdun_asp', type=float, default=5.0, help='')
   addarg('--maxdun_glu', type=float, default=5.0, help='')
   addarg('--maxdun_his', type=float, default=5.0, help='')
   addarg('--scale_number_of_rotamers', type=float, default=1.0,
          help='modify resolution of rotamers')
   addarg('--output_prefix', type=str, default='results/mofdock_')
   addarg('--max_2res_score', type=float, default=10.0)
   parser.has_mof_args = True
   return parser

def get_cli_args(argv=None, parent=None, **kw):
   parser = default_cli_parser(parent, **kw)
   argv = sys.argv[1:] if argv is None else argv
   argv = rp.app.options.make_argv_with_atfiles(argv, **kw)
   options = parser.parse_args(argv)
   # options = process_cli_args(options, **kw)
   return rp.Bunch(options)

def defaults():
   return get_cli_args([])
