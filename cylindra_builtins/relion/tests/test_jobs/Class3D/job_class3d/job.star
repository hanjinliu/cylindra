
# version 50001

data_job

_rlnJobTypeLabel             relion.class3d
_rlnJobIsContinue                       0
_rlnJobIsTomo                           1


# version 50001

data_joboptions_values

loop_
_rlnJobOptionVariable #1
_rlnJobOptionValue #2
allow_coarser         No
ctf_intact_first_peak         No
do_apply_helical_symmetry        Yes
  do_blush         No
do_combine_thru_disc         No
do_ctf_correction        Yes
do_fast_subsets         No
  do_helix         No
do_local_ang_searches        Yes
do_local_search_helical_symmetry         No
   do_pad1         No
do_parallel_discio        Yes
do_preread_images         No
  do_queue         No
do_zero_mask        Yes
dont_skip_align        Yes
   fn_cont         ""
   fn_mask         ""
    fn_ref Refine3D/job015/run_class001.mrc
   gpu_ids         ""
helical_nr_asu          1
helical_range_distance         -1
helical_rise_inistep          0
helical_rise_initial          0
helical_rise_max          0
helical_rise_min          0
helical_tube_inner_diameter         -1
helical_tube_outer_diameter         -1
helical_twist_inistep          0
helical_twist_initial          0
helical_twist_max          0
helical_twist_min          0
helical_z_percentage         30
highres_limit         -1
in_optimisation         ""
in_particles XXX
in_tomograms YYY
in_trajectories         ""
  ini_high         60
keep_tilt_prior_fixed        Yes
min_dedicated         24
nr_classes          9
   nr_iter         25
    nr_mpi          5
   nr_pool         30
nr_threads          6
offset_range          5
offset_step          1
other_args         ""
particle_diameter        230
      qsub     sbatch
qsubscript /public/EM/RELION/relion-slurm-gpu-4.0.csh
 queuename    openmpi
 range_psi         10
 range_rot         -1
range_tilt         15
ref_correct_greyscale        Yes
 relax_sym         ""
  sampling "1.8 degrees"
scratch_dir $RELION_SCRATCH_DIR
sigma_angles          5
sigma_tilt         10
  sym_name         C1
 tau_fudge          1
trust_ref_size        Yes
use_direct_entries        Yes
   use_gpu        Yes
