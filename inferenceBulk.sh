#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                   
#SBATCH --qos=mig_class
#SBATCH --output=/home/en540-lludwig2/ProMDLM/inference.log
#SBATCH --time=00:10:00
#SBATCH --export=ALL

module load miniconda3

source /data/apps/extern/spack_on/gcc/9.3.0/miniconda3/22.11.1-7f5s6r5uqyngliaca4moeawkxnnsmwkq/etc/profile.d/conda.sh

conda activate CompClass

cd /home/en540-lludwig2/ProMDLM
python generate_bulk.py
