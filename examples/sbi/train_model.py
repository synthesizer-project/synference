import datetime
import multiprocessing as mp
import sys
from ast import literal_eval
from dataclasses import dataclass
import os

import torch
from astropy.table import Table
from simple_parsing import ArgumentParser

from sbifitter import SBI_Fitter, create_uncertainity_models_from_EPOCHS_cat

try:
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    print("Multiprocessing start method set to 'spawn'.")
except RuntimeError as e:
    print(f"Start method already set: {e}")


# Setup parsing
parser = ArgumentParser(description="SBI SED Fitting")

file_dir = os.path.dirname(__file__)


@dataclass
class Args:
    learning_rate: float = 1e-4
    stop_after_epochs: int = 20
    training_batch_size: int = 64
    validation_fraction: float = 0.1
    clip_max_norm: float = 5.0
    backend: str = "sbi"
    engine: str = "NPE"
    model_types: str = "maf"
    n_nets: int = 1
    model_name: str = "BPASS_Chab_DelayedExpSFH_0.01_z_12_CF00_v1"
    name_append: str = ""
    grid_path: str = f"{file_dir}/../../grids/grid_BPASS_Chab_LogNormal_SFH_0.001_z_12_logN_5.0_Calzetti_v2.hdf5" # noqa
    hidden_features: int = 64
    num_transforms: int = 6
    num_components: int = 10
    scatter_fluxes: float = 10.0
    include_errors_in_feature_array: bool = True
    norm_mag_limit: float = 40.0
    drop_dropouts: bool = True
    drop_dropout_fraction: float = 0.5
    plot: bool = True
    additional_model_args: tuple = ()
    data_err_file: str = """/home/tharvey/Downloads/JADES-Deep-GS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV.fits"""  # noqa
    data_err_hdu: str = 'OBJECTS'  # The HDU name in the FITS file
    background: bool = False
    model_features: tuple = ()
    norm_method: str = None
    device: str = "cuda:0"


parser.add_arguments(Args, dest="args")


def parse_args():
    args, _ = parser.parse_known_args()
    print(args)
    return args.args


def main_task(args: Args) -> None:
    """
    Main logic for the SED fitting task.
    This function can be run in a separate process.
    """
    # If running in the background, redirect output to log files.
    if args.background:
        print(
            """Background process started.
            Logging to sbi_training.log and sbi_training_error.log"""
        )
        sys.stdout = open("sbi_training.log", "a", buffering=1)
        sys.stderr = open("sbi_training_error.log", "a", buffering=1)
        print(f"Training started at {datetime.datetime.now()}", file=sys.stdout)
        print(f"Arguments: {args}", file=sys.stdout)

    if args.scatter_fluxes > 0:
        
        table = Table.read(args.data_err_file, format="fits", hdu=args.data_err_hdu)
        bands = [i.split("_")[-1] for i in table.colnames if i.startswith("loc_depth")]
        hst_bands = ['F435W', 'F606W','F775W', 'F814W', 'F850LP']
        new_band_names = [f"HST/ACS_WFC.{band.upper()}" if band in hst_bands else
            f"JWST/NIRCam.{band.upper()}" for band in bands]

        empirical_noise_models = create_uncertainity_models_from_EPOCHS_cat(
            args.data_err_file, bands, new_band_names, plot=False, hdu=args.data_err_hdu,
        )
    else:
        empirical_noise_models = {}


    additional_model_args = {}
    if args.additional_model_args:
        # Assuming args.additional_model_args is a string like 'key1=val1,key2=val2'
        try:
            extra_model_args_str = args.additional_model_args[0]
            temp = {}
            for arg in extra_model_args_str.split(","):
                key, value = arg.split("=")
                temp[key.strip()] = literal_eval(value.strip())
            additional_model_args = temp
        except (ValueError, SyntaxError) as e:
            print(
                f"Warning: Could not parse additional_model_args. Error: {e}",
                file=sys.stderr,
            )

    if len(additional_model_args) > 0:
        print("Additional model args:", additional_model_args)

    empirical_model_fitter = SBI_Fitter.init_from_hdf5(
        model_name=args.model_name, hdf5_path=args.grid_path, device=args.device
    )

    unused_filters = [
        filt
        for filt in empirical_model_fitter.raw_photometry_names
        if filt not in list(empirical_noise_models.keys())
    ]



    empirical_model_fitter.create_feature_array_from_raw_photometry(
        extra_features=list(args.model_features),
        normalize_method=args.norm_method,
        include_errors_in_feature_array=args.include_errors_in_feature_array,
        scatter_fluxes=args.scatter_fluxes,
        empirical_noise_models=empirical_noise_models
        if args.include_errors_in_feature_array
        else None,
        photometry_to_remove=unused_filters if args.include_errors_in_feature_array else None,
        norm_mag_limit=args.norm_mag_limit,
        drop_dropouts=args.drop_dropouts,
        drop_dropout_fraction=args.drop_dropout_fraction,
        parameters_to_add=['mwa'],
    )

    #col_i = empirical_model_fitter.feature_array[:, 0]
    #n = len(col_i)
    #import numpy as np
    #v25, v75 = np.percentile(col_i, [25, 75])
    #print(v25, v75, dx, n, 'check')
    #dx = 2 * (v75 - v25) / (n ** (1 / 3))
    #empirical_model_fitter.plot_histogram_feature_array()
    empirical_model_fitter.plot_histogram_parameter_array()


    empirical_model_fitter.run_single_sbi(
        n_nets=args.n_nets,
        backend=args.backend,
        engine=args.engine,
        stop_after_epochs=args.stop_after_epochs,
        learning_rate=args.learning_rate,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        num_components=args.num_components,
        model_type=args.model_types,
        training_batch_size=args.training_batch_size,
        validation_fraction=args.validation_fraction,
        clip_max_norm=args.clip_max_norm,
        name_append=args.name_append,
        plot=args.plot,
        additional_model_args=additional_model_args,
    )

    print(f"Training finished at {datetime.datetime.now()}", file=sys.stdout)


if __name__ == "__main__":
    args = parse_args()

    if args.background:
        # Use multiprocessing to run the main task in a new 'spawned' process
        print("Starting the task in a background process...")
        process = mp.Process(target=main_task, args=(args,))
        process.start()
        print(f"Process started with PID: {process.pid}. The script will now exit.")
        sys.exit(0)
    else:
        # Run in the foreground as normal
        main_task(args)
