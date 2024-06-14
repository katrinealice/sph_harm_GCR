Statistical estimation of full-sky radio maps from 21cm array visibility data using Gaussian Constrained Realisations

Stand-alone version of code otherwise included in HYDRA. 

This repo contains the shell script and python script to generate Gaussian Constrained Realisations of the spherical harmonics modes of the sky given some visibility data and prior. This version uses simulated data and parameters are set as in the 'standard case' in our paper https://arxiv.org/abs/2403.13766. 

Some command line arguments can be set:
-dir: the output directory name 
-nsamples: how many samples you want to generate
-data_seed: the random seed for the noise realisation on the simulated data
-jobid: Default $SLURM_ARRAY_TASK_ID. This also determines the random seed for the constrained realisations (along with the specific sample number)

Additionally
If you want fx 4000 samples, simply set the --array parameter to make copies of the shell script, --array=1-40 with -nsamlpes=100

Not yet implemented:
More command-line arguments for the different parameters should be included in the future, for now they have to be changed manually in the python script. 
