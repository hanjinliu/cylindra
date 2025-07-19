# version 50001

data_job

_rlnJobTypeLabel             relion.refine3d.tomo
_rlnJobIsContinue                       0
_rlnJobIsTomo                           1


# version 50001

data_joboptions_values

loop_
_rlnJobOptionVariable #1
_rlnJobOptionValue #2
auto_faster         No
auto_local_sampling "3.7 degrees"
ctf_intact_first_peak         No
do_apply_helical_symmetry        Yes
  do_blush         No
do_combine_thru_disc         No
do_ctf_correction        Yes
  do_helix         No
do_local_search_helical_symmetry         No
   do_pad1         No
do_parallel_discio        Yes
do_preread_images         No
  do_queue         No
do_solvent_fsc         No
do_zero_mask        Yes
   fn_cont         ""
   fn_mask         ""
    fn_ref InitialModel/job_initial_model/initial_model.mrc
   gpu_ids          0
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
in_optimisation InitialModel/job_initial_model/run_it100_optimisation_set.star
in_particles         ""
in_tomograms         ""
in_trajectories         ""
  ini_high         60
keep_tilt_prior_fixed        Yes
min_dedicated          1
    nr_mpi          3
   nr_pool          3
nr_threads          1
offset_range          5
offset_step          1
other_args         ""
particle_diameter        550
      qsub     sbatch
qsubscript /path/to/relion_qsub.csh
 queuename    openmpi
 range_psi         10
 range_rot         -1
range_tilt         15
ref_correct_greyscale         No
 relax_sym         ""
  sampling "15 degrees"
scratch_dir         ""
sigma_tilt         -1
  sym_name         C1
trust_ref_size        Yes
use_direct_entries         No
   use_gpu        Yes
