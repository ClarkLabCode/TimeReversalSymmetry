#!/bin/bash

module load dSQ

dsq --job-file=joblist_natural_phase_randomized.txt \
--job-name=phase_randomized_model_natural \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
--mem-per-cpu=15G \
--time=1- \
--mail-type=ALL \
--mail-user=baohua.zhou@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist_natural_phase_randomized-%A_%a-%N.out