#!/bin/bash -l
#SBATCH --job-name=sbi_training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=20
#SBATCH -t 10:00:00
#SBATCH -p cosma7
#SBATCH -A dp276
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.harvey-3@manchester.ac.uk
#SBATCH --output=./logs/log%A_%a.out
#SBATCH --error=./logs/log%A_%a.err


# Set the directory
cd /cosma7/data/dp276/dc-harv3/work/

module load python/3.12.4 gnu_comp/11.1.0 openmpi
source venv_ltu_ili/bin/activate

echo $PWD

python synference/examples/sbi/train_model.py $1
