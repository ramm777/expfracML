#!/bin/sh
#Run the job from this directory (-cwd)
#$ -cwd
#Keep all my current environmental variables (-V)
#$ -V
module load matlab/R2019b
matlab -nodisplay -nosplash < runmodel.m > results.log