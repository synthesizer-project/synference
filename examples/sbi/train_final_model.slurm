#!/bin/bash -l
#SBATCH --job-name=missing_fluxes_just_z_training_final
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

export GRID="BPASS"

alias mysql=/cosma/apps/dp276/dc-harv3/mysql/bin/mysql
export LD_LIBRARY_PATH="/cosma/apps/dp276/dc-harv3/mysql/lib:$LD_LIBRARY_PATH"

# Force the library to be loaded at startup to fix the TLS error
export LD_PRELOAD="/cosma/apps/dp276/dc-harv3/mysql/lib/libmysqlclient.so.24"
#srun python synference/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,w,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4 --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_${GRID}_Chab_DenseBasis_SFH_0.01_z_14_logN_6.0_Calzetti_v3_multinode.hdf5 --model_name=DenseBasis_v3_final_final --name_append=nsf --train_test_fraction=0.90 --parameter_transformations="Av=log10,sfr_10=log10_floor,mass_weighted_age=log10" --parameters_to_add="mass_weighted_age,sfr_10,log_surviving_mass,beta" --noise_model_clas=asinh --model_types=nsf --model_features='redshift' --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params.yaml --device=cpu
# just redshift

#srun python synference/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,w,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4 --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_${GRID}_Chab_DenseBasis_SFH_0.01_z_14_logN_6.0_Calzetti_v4_multinode.hdf5 --model_name=${GRID}_DenseBasis_v4_final --name_append=nsf_${SLURM_NODEID} --train_test_fraction=0.90 --parameter_transformations="Av=log10,sfr_10=log10_floor,mass_weighted_age=log10" --parameters_to_add="mass_weighted_age,sfr_10,log_surviving_mass,beta" --noise_model_clas=asinh --model_types=nsf --model_features='redshift' --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params_2101.yaml --device=cpu
# just redshift

# python synference/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4 --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_BPASS_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v3_multinode.hdf5 --model_name=DenseBasis_v3_just_redshift --name_append=nsf --train_test_fraction=0.90 --noise_model_clas=asinh --num_transforms=15 --hidden_features=69 --learning_rate=0.00029182522784313434 --batch_size=52 --stop_after_epochs=47 --clip_max_norm=4.778606871366431 --model_types=nsf --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params.yaml --device=cuda --parameters_to_remove=log_mass,Av,sfh_quantile_25,sfh_quantile_50,sfh_quantile_75,log_sfr,log10metallicity


srun python synference/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,w,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4 --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_${GRID}_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v4_multinode.hdf5 --model_name=${GRID}_DenseBasis_v4_final_missing_bands_just_redshift --name_append=nsf --train_test_fraction=0.90 --noise_model_clas=asinh --model_types=nsf --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params_2101.yaml --device=cpu --simulate_missing_fluxes=True --include_flags_in_feature_array=True --missing_flux_options="none,F335M,F453W,F606W,F775W,F814W,F850LP,F435W+F606W+F775W+F814W+F850LP" --parameters_to_remove=log_mass,Av,sfh_quantile_25,sfh_quantile_50,sfh_quantile_75,log_sfr,log10metallicity




srun python synference/examples/sbi/scripts/train_model.py --scatter_fluxes=0 --include_errors_in_feature_array=False --set_photometry=CFHT/MegaCam.u,Euclid/VIS.vis,Euclid/NISP.Y,Euclid/NISP.J,Euclid/NISP.H,Subaru/HSC.Y,Subaru/HSC.z,Subaru/HSC.i,Subaru/HSC.g,Subaru/HSC.r,Paranal/VISTA.Ks,Paranal/VISTA.H,Paranal/VISTA.J,Paranal/VISTA.Y,Spitzer/IRAC.I1,Spitzer/IRAC.I2 --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_${GRID}_Chab_DenseBasis_SFH_0.01_z_14_logN_5.0_Calzetti_v4_multinode.hdf5 --model_name=${GRID}_DenseBasis_v4_Euclid_Deep --name_append=nsf --train_test_fraction=0.90 --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params_2101.yaml --device=cuda --background
echo -e "\nJob finished."


run python synference/examples/sbi/scripts/train_model.py --data_err_file="/cosma7/data/dp276/dc-harv3/work/catalogs/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits" --scatter_fluxes=1 --include_errors_in_feature_array=True --photometry_to_remove=F070W,F140M,F090W,F162M,F182M,F210M,F250M,F300M,F360M,F430M,F460M,F480M,F475W,Z,Y,J,H,Ks,g,r,i,z,Y,u,g,r,i,w,z,vis,Y,J,H,F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W,I1,I2,I3,I4,F435W,F775W,F850LP --grid_path=/cosma/apps/dp276/dc-harv3/synference/grids/grid_${GRID}_Chab_DenseBasis_SFH_0.01_z_14_logN_6.0_Calzetti_v4_multinode.hdf5 --model_name=${GRID}_DenseBasis_v4_final_ceers --name_append=nsf --train_test_fraction=0.90 --noise_model_clas=asinh --model_types=nsf --custom_config_yaml=/cosma/apps/dp276/dc-harv3/synference/examples/sbi/best_params_2101.yaml --device=cuda --background 