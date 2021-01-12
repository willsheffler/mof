import sys, argparse, rpxdock as rp

def default_cli_parser(parent=None, **kw):
   parser = parent if parent else argparse.ArgumentParser(allow_abbrev=False)
   addarg = rp.app.options.add_argument_unless_exists(parser)

   # yapf: disable

   addarg("inputs", nargs="*", type=str, default=[], help='input structures')

   addarg('--aa_labels', type=str, nargs='*',                   default='CYS DCYS ASP DASP GLU DGLU HIS DHIS HISD DHISD'.split(),
          help='choices: CYS DCYS ASP DASP GLU DGLU HIS DHIS HISD DHISD',)
   addarg('--aa_pair_labels', type=str, nargs='*',              default=['ALL'],             help='give in pairs (--aa_pair_labels A B C D yeilds A-B and C-B pairs) choices: CYS DCYS ASP DASP GLU DGLU HIS DHIS HISD DHISD')
   addarg('--angle_err_tolerance', type=float,                  default=15,                  help='max allowed angular deviation from ideal metal binding. applied early, so ok to be generous')
   addarg('--angle_to_cart_err_ratio', type=float,              default=20.0,                help='lever distance to equate anglur and cartesian errors. probably no reason to change, unless you know why')
   addarg('--chiresl_asp1', type=float,                         default=8.0,                 help='resolution of scanning for asp chi1')
   addarg('--chiresl_asp2', type=float,                         default=5.0,                 help='resolution of scanning for asp chi2')
   addarg('--chiresl_cys1', type=float,                         default=6.0,                 help='resolution of scanning for cys chi1')
   addarg('--chiresl_cys2', type=float,                         default=8.0,                 help='resolution of scanning for cys chi2')
   addarg('--chiresl_glu1', type=float,                         default=6.0,                 help='resolution of scanning for glu chi1')
   addarg('--chiresl_glu2', type=float,                         default=12.0,                help='resolution of scanning for glu chi2')
   addarg('--chiresl_glu3', type=float,                         default=6.0,                 help='resolution of scanning for glu chi3')
   addarg('--chiresl_his1', type=float,                         default=3.0,                 help='resolution of scanning for his chi1')
   addarg('--chiresl_his2', type=float,                         default=8.0,                 help='resolution of scanning for his chi2')
   addarg('--clash_dis', type=float,                            default=3.3,                 help='distance below which atoms "clash"')
   addarg('--contact_dis', type=float,                          default=7.0,                 help='max CB-CB distance between residue "neighbors"')
   # addarg('--continue_from_checkpoints', action="store_true", default=False,               help='')
   addarg('--cst_ang_metal', type=float,                        default=109.47,              help='desired angle between metal liganding atoms')
   addarg('--cst_dis_metal', type=float,                        default=2.2,                 help='desired distance from metal to liganding atoms')

   addarg('--cst_sd_cut_ang', type=float,                       default=0.01,                help='std dev of angluar cutpoint constraint (lower is stronger constraint)')
   addarg('--cst_sd_cut_dih', type=float,                       default=0.1,                 help='std dev of dihedral cutpoint constraint (lower is stronger constraint)')
   addarg('--cst_sd_cut_dis', type=float,                       default=0.01,                help='std dev of distance cutpoint constraint (lower is stronger constraint)')

   addarg('--cst_sd_metal_coo', type=float,                     default=0.5,                 help='std dev of metal-O-C-O dihedral constraint (lower is stronger constraint)')
   addarg('--cst_sd_metal_dir', type=float,                     default=0.4,                 help='std dev of ligand "orbital points at metal" angle constraint (lower is stronger constraint)')
   addarg('--cst_sd_metal_lig_ang', type=float,                 default=0.4,                 help='std dev of lig-metal-lig angle constraint (lower is stronger constraint)')
   addarg('--cst_sd_metal_lig_dist', type=float,                default=0.2,                 help='std dev of metal-lig distance constraint (lower is stronger constraint)')
   addarg('--cst_sd_metal_olap', type=float,                    default=0.03,                help='std dev of distance between symmeric copies of metal constraint (lower is stronger constraint)')

   addarg('--test_run', action="store_true",                       default=False,               help='ignores most flags and inputs, set to test values')
   addarg('--debug', action="store_true",                       default=False,               help='extra output and maybe modified behavoir')
   addarg('--dist_err_tolerance', type=float,                   default=1.0,                 help='max allowed deviation of symmetric metal overlap from 0. applied early, so ok to be generous')
   addarg("--dont_replace_these_aas", nargs="*", type=str,      default=['PRO'],             help='AAs which should not be changed')
   addarg('--err_tolerance', type=float,                        default=2.0,                 help='max allowed combination of dist_err and angle_err. applied early, so ok to be generous')
   addarg('--max_2res_score', type=float,                       default=10.0,                help='max score delta upon placing rotamers')
   addarg('--max_bb_redundancy', type=float,                    default=0.1,                 help='max non-algned rms distance between any outputs')
   addarg('--max_cell_size', type=float,                        default=50,                  help='maximum cell size (will be correlated to peptide size, maybe use --max_solv_frac if you have mixed size peptides')
   addarg('--max_dun_score', type=float,                        default=6.0,                 help='overall maximum DUN score ever allowed for consideration')
   addarg('--max_pept_size', type=int,                          default=10,                  help='reserve output space for this many residues (keep same to match run results together)')
   addarg('--max_score_minimized', type=float,                  default=50.0,                help='maximum score after minimization (including cst and all)')
   addarg('--max_solv_frac', type=float,                        default=0.8,                 help='maximum solvent fraction acceted. this is highly approximate... recommend spot-checking and adjusting as needed')
   addarg('--max_sym_score', type=float,                        default=100.0,               help='max nonbonded energy across xtal contacts (induding clash)')
   addarg('--maxdun_asp', type=float,                           default=5.0,                 help='max dunbrak score for asp')
   addarg('--maxdun_cys', type=float,                           default=4.0,                 help='max dunbrak score for cys')
   addarg('--maxdun_glu', type=float,                           default=5.0,                 help='max dunbrak score for glu')
   addarg('--maxdun_his', type=float,                           default=5.0,                 help='max dunbrak score for his')
   addarg('--min_cell_size', type=float,                        default=0,                   help='minimum cell size')
   addarg('--min_contacts', type=float,                         default=0,                   help='minimun number of res-res contacts across symmetric units')
   addarg('--min_dist_to_z_axis', type=float,                   default=5.0,                 help='maybe dont change this')
   addarg('--output_prefix', type=str,                          default='results/mofdock_',  help='prefix to output filenames')
   addarg('--overwrite', action="store_true",                   default=False,               help='')
   addarg('--rotcloud_cache', type=str,                         default='.rotcloud_cache',   help='cache file location')
   addarg('--scale_number_of_rotamers', type=float,             default=1.0,                 help='modify resolution of rotamers')
   addarg('--sfxn_minimize_weights', type=str,                  default='minimize.wts',      help='minimization score func wts file')
   addarg('--sfxn_rotamer_weights', type=str,                   default='rotamer.wts',       help='rotamer scanning score func wts file')
   addarg('--sfxn_sterics_weights', type=str,                   default='sterics.wts',       help='clash checking score func wts file')
   addarg('--spacegroups', nargs='*', type=str,                 default=[],                  help='list of spacegroups')
   addarg('--sym_axes_angle_tolerance', type=float,             default=5.0,                 help='max deviation from crystal "magic angle"')

   # yapf: enable

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
