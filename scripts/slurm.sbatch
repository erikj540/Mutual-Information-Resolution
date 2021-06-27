#!/bin/bash

##SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:50:00
#SBATCH --mem-per-cpu=500MB

# mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erjo3868@colorado.edu

#SBATCH --job-name=mi
#SBATCH --output=/Users/erjo3868/Bayesian-Vaccine-Efficacy/fiji_out/mi.out
#SBATCH --error=/Users/erjo3868/Mutual-Information-Resolution/fiji_out/mi.err

/Users/erjo3868/anaconda3/envs/stan/bin/python /Users/erjo3868/Bayesian-Vaccine-Efficacy/allocating_validation_tests/run_sims.py -path=${path}