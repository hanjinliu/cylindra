
# version 50001

data_job

_rlnJobTypeLabel             relion.picktomo
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0


# version 50001

data_joboptions_values

loop_
_rlnJobOptionVariable #1
_rlnJobOptionValue #2
  do_queue         No
in_star_file         ""
in_tomoset Tomograms/job_tomo/tomograms.star
min_dedicated          1
other_args         ""
particle_spacing         -1
 pick_mode  particles
      qsub     sbatch
qsubscript /path/to/relion/bin/relion_qsub.csh
 queuename    openmpi
