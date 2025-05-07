

Notes explaining use cases of scripts.

For Python environment do 
source /cosma7/data/dp276/dc-harv3/work/venv_ltu_ili/activate

generate_grid_lhc.py 
    - Defines SFH/SPS model from chosen grid. Includes two epochs of SFH, including
    one standard BPASS SPS, and one Ygrdrassil Pop III model.
    - Draws priors from LatinHypercube, generates spectra/photometry using Synthesizer.
    - Combines models together to produce full grid.

generate_grid_lhc_standard_sps.py
    - Very similar to generate_grid_lhc.py, but only includes a Pop I/Pop II BPASS component.

grab_filters_for_node.py
    - Generates NIRCam filter HDF5 to load in nodes, since they don't have Web access. Only needed
    if running on HPC cluster.

make_grid.slurm
    - Submits generate_grid_lhc.py to a Slurm queue. 

train_model.py 
    - Runs SBI on saved grid.

Various other notebooks for testing etc. 


