#!/bin/bash -l
#SBATCH --job-name=sbi_training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=20
#SBATCH -t 72:00:00
#SBATCH -p cosma7
#SBATCH -A dp276
#SBATCH --mail-type=ALL
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

echo "### STEP 1: Starting parallel work on all nodes. ###"
echo "Node ${SLURM_NODEID}: Running Optuna optimization script."

alias mysql=/cosma/apps/dp276/dc-harv3/mysql/bin/mysql
export LD_LIBRARY_PATH="/cosma/apps/dp276/dc-harv3/mysql/lib:$LD_LIBRARY_PATH"

# Force the library to be loaded at startup to fix the TLS error
export LD_PRELOAD="/cosma/apps/dp276/dc-harv3/mysql/lib/libmysqlclient.so.24"

srun python sbifitter/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4 --grid_path=/cosma/apps/dp276/dc-harv3/sbifitter/grids/grid_BPASS_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v3_multinode.hdf5 --model_name=DenseBasis_v3_final_custom_small2 --name_append=nsf_zfix --train_test_fraction=0.90 --parameter_transformations="Av=log10,sfr_10=log10_floor,mass_weighted_age=log10" --parameters_to_add="mass_weighted_age,sfr_10,log_surviving_mass,beta" --noise_model_clas=asinh --num_transforms=15 --hidden_features=69 --learning_rate=0.00029182522784313434 --batch_size=52 --stop_after_epochs=47 --clip_max_norm=4.778606871366431 --model_types=nsf --model_features='redshift' --custom_config_yaml=/cosma/apps/dp276/dc-harv3/sbifitter/examples/sbi/best_params.yaml --device=cpu
echo -e "\nJob finished."
