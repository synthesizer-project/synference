#!/bin/bash -l
#SBATCH --job-name=sbi_grid_generation
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -t 12:00:00
#SBATCH -p cosma7
#SBATCH -A dp276
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=thomas.harvey-3@manchester.ac.uk
#SBATCH --output=/cosma/apps/dp276/dc-harv3/slurm_logs/log%A_%a.out
#SBATCH --error=/cosma/apps/dp276/dc-harv3/slurm_logs/log%A_%a.err

# Set the directory
cd /cosma/apps/dp276/dc-harv3/

module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3
source venv_simformer/bin/activate

echo "Current directory: $PWD"
echo "Job running on nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of CPUs per task: $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create a unique directory on a shared filesystem for "done" files
DONE_DIR="/cosma/apps/dp276/dc-harv3/temp/${SLURM_JOB_ID}_done_files"
mkdir -p "$DONE_DIR"
echo "### STEP 1: Starting parallel work on all nodes. ###"
echo "Node ${SLURM_NODEID}: Running grid generation script."
mpirun python sbifitter/examples/grid_generation/final_grid_generation_multinode.py $SLURM_CPUS_PER_TASK 0
if [ $? -ne 0 ]; then
  echo "ERROR: One or more tasks in the parallel step failed. Aborting job." >&2
  # Clean up the temporary directory before exiting
  rm -rf "$DONE_DIR"
  exit 1
fi
echo "Node ${SLURM_NODEID}: Script finished, creating done file."
srun bash -c 'touch "'"$DONE_DIR"'/node_${SLURM_NODEID}.done"'
echo -e "\n### STEP 2: Starting final task on primary node after barrier. ###"

#srun --nodes=1 --ntasks=1 python sbifitter/examples/grid_generation/final_grid_generation_multinode.py $SLURM_CPUS_PER_TASK 1


srun --nodes=1 --ntasks=1 --exact bash -c '
  echo "Primary (Node 0): Waiting for all nodes to complete Step 1..."
  while [ $(ls "'"$DONE_DIR"'" | wc -l) -lt '"$SLURM_NNODES"' ]; do
    sleep 2
  done
  echo "Primary (Node 0): All nodes are done. Running final aggregation script."
  python sbifitter/examples/grid_generation/final_grid_generation_multinode.py $SLURM_CPUS_PER_TASK 1
'

#rm -rf "$DONE_DIR"
echo -e "\nJob finished."
