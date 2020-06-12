#!/bin/bash
#
#SBATCH --job-name=skeldump_conv_2017
#
#SBATCH --ntasks=1 --cpus-per-task=40 --mem=40gb

cd /mnt/rds/redhen/gallina
module load singularity

srun singularity exec shub://frankier/gsoc2020:skeldump python \
    /opt/redhen/skeldump/skeldump.py \
    conv --cores 40 --mode BODY_25_HANDS \
    monolithic-tar projects/2017_openpose_body_hand.tar home/frr7/openpose2017