#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=120G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-08:00
#SBATCH -o /nfs/sloanlab001/projects/ssteffen_proj/bgt_occ_change/output/log.out
#SBATCH -e /nfs/sloanlab001/projects/ssteffen_proj/bgt_occ_change/output/errors.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ssteffen@mit.edu
#SBATCH --job-name="BGT Job0"

module load engaging/anaconda/2.3.0
source activate py36
xvfb-run -d python3 bgt_wages_management.py
