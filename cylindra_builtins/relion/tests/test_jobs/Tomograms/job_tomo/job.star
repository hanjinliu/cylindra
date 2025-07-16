# version 50001

data_job

_rlnJobTypeLabel             relion.reconstructtomograms
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0


# version 50001

data_joboptions_values

loop_
_rlnJobOptionVariable #1
_rlnJobOptionValue #2
binned_angpix         15
centre_proj          0
ctf_intact_first_peak        Yes
do_fourier         No
   do_proj         No
  do_queue         No
generate_split_tomograms         No
in_tiltseries AlignTiltSeries/job0XX/aligned_tilt_series.star
min_dedicated          1
    nr_mpi          1
nr_threads          4
other_args         ""
      qsub     sbatch
qsubscript /path/to/relion/bin/relion_qsub.csh
 queuename    openmpi
thickness_proj         10
tiltangle_offset          0
 tomo_name         ""
      xdim       4000
      ydim       4000
      zdim        800
