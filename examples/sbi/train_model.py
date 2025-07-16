import datetime
import multiprocessing as mp
import os
import sys
from ast import literal_eval
from dataclasses import dataclass

import numpy as np
import torch
from astropy.table import Table
from simple_parsing import ArgumentParser

from sbifitter import SBI_Fitter, Simformer_Fitter, create_uncertainity_models_from_EPOCHS_cat

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
    train_test_fraction: float = 0.9
    validation_fraction: float = 0.1
    clip_max_norm: float = 5.0
    backend: str = "sbi"
    engine: str = "NPE"
    model_types: str = "maf"
    n_nets: int = 1
    model_name: str = "sbi_model"
    name_append: str = ""
    grid_path: str = (
        f"{file_dir}/../../grids/grid_BPASS_Chab_LogNormal_SFH_0.001_z_12_logN_5.0_Calzetti_v2.hdf5"  # noqa
    )
    hidden_features: int = 64
    num_transforms: int = 6
    num_components: int = 10
    scatter_fluxes: float = 10.0
    include_errors_in_feature_array: bool = True
    norm_mag_limit: float = 40.0
    drop_dropouts: bool = True
    drop_dropout_fraction: float = 0.5
    max_rows: int = -1
    parameter_transformations: tuple = ()  # This can be a dict of transformations
    photometry_to_remove: tuple = ()
    plot: bool = True
    additional_model_args: tuple = ()
    parameters_to_add: tuple = ()
    data_err_file: str = """/home/tharvey/Downloads/JADES-Deep-GS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV.fits"""  # noqa
    data_err_hdu: str = "OBJECTS"  # The HDU name in the FITS file
    background: bool = False
    model_features: tuple = ()
    norm_method: str = None
    device: str = "cuda:0"
    simformer: bool = False  # If True, use SimFormer for training


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

    phot_to_remove = args.photometry_to_remove[0]
    phot_to_remove = phot_to_remove.split(",")
    hst_bands = ["F435W", "F606W", "F775W", "F814W", "F850LP"]

    if args.scatter_fluxes > 0:
        table = Table.read(args.data_err_file, format="fits", hdu=args.data_err_hdu)
        bands = [i.split("_")[-1] for i in table.colnames if i.startswith("loc_depth")]
        if len(phot_to_remove) > 0:
            bands = [band for band in bands if band not in phot_to_remove]

        new_band_names = [
            f"HST/ACS_WFC.{band.upper()}" if band in hst_bands else f"JWST/NIRCam.{band.upper()}"
            for band in bands
        ]

        root_out_dir = os.path.dirname(os.path.dirname(file_dir))
        out_dir = f"{root_out_dir}/models/{args.model_name}/plots/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        empirical_noise_models = create_uncertainity_models_from_EPOCHS_cat(
            args.data_err_file,
            bands,
            new_band_names,
            plot=True,
            hdu=args.data_err_hdu,
            save_path=out_dir,
            save=True,
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
    parameter_transformations = {}

    if args.parameter_transformations:
        pos_vals = {"log10": np.log10, "log": np.log, "exp": np.exp, "sqrt": np.sqrt}
        # Assuming args.parameter_transformations is a string like 'key1=val1,key2=val2'
        try:
            transformations_str = args.parameter_transformations[0]
            temp = {}
            for arg in transformations_str.split(","):
                key, value = arg.split("=")
                value = value.strip()
                if value in pos_vals:
                    temp[key.strip()] = pos_vals[value]
                else:
                    try:
                        # Attempt to evaluate as a Python literal
                        temp[key.strip()] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        # If it fails, treat it as a string
                        temp[key.strip()] = value.strip()
            parameter_transformations = temp
        except (ValueError, SyntaxError) as e:
            print(
                f"Warning: Could not parse parameter_transformations. Error: {e}",
                file=sys.stderr,
            )

    if len(additional_model_args) > 0:
        print("Additional model args:", additional_model_args)

    if args.simformer:
        fitter_class = Simformer_Fitter
    else:
        fitter_class = SBI_Fitter

    empirical_model_fitter = fitter_class.init_from_hdf5(
        model_name=args.model_name, hdf5_path=args.grid_path, device=args.device
    )

    if not args.include_errors_in_feature_array:
        unused_filters = [
            filt
            for filt in empirical_model_fitter.raw_photometry_names
            if filt not in list(empirical_noise_models.keys())
        ]
    else:
        unused_filters = [
            f"JWST/NIRCam.{filt.upper()}"
            if filt not in hst_bands
            else f"HST/ACS_WFC.{filt.upper()}"
            for filt in phot_to_remove
        ]
        unused_filters = [
            filt for filt in unused_filters if filt in empirical_model_fitter.raw_photometry_names
        ]

    print(f"Unused filters: {unused_filters}", file=sys.stdout)
    empirical_model_fitter.create_feature_array_from_raw_photometry(
        extra_features=list(args.model_features),
        normalize_method=args.norm_method,
        include_errors_in_feature_array=args.include_errors_in_feature_array,
        scatter_fluxes=args.scatter_fluxes,
        empirical_noise_models=empirical_noise_models
        if args.include_errors_in_feature_array
        else None,
        photometry_to_remove=unused_filters,
        norm_mag_limit=args.norm_mag_limit,
        drop_dropouts=args.drop_dropouts,
        drop_dropout_fraction=args.drop_dropout_fraction,
        parameters_to_add=args.parameters_to_add,
        parameter_transformations=parameter_transformations,
        max_rows=args.max_rows,
    )

    # col_i = empirical_model_fitter.feature_array[:, 0]
    # n = len(col_i)
    # import numpy as np
    # v25, v75 = np.percentile(col_i, [25, 75])
    # print(v25, v75, dx, n, 'check')
    # dx = 2 * (v75 - v25) / (n ** (1 / 3))
    empirical_model_fitter.plot_histogram_feature_array(bins="scott")
    empirical_model_fitter.plot_histogram_parameter_array(bins="scott")

    if not args.simformer:
        args = dict(
            train_test_fraction=args.train_test_fraction,
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
    else:
        args = dict(
            backend="jax",
            num_training_simulations=10_000,
            train_test_fraction=args.train_test_fraction,
            random_seed=42,
            set_self=True,
            verbose=True,
            load_existing_model=True,
            name_append=args.name_append,
            save_method="joblib",
            task_func=None,
            model_config_dict={
                "name": "ScoreTransformer",
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 4,
                "d_feedforward": 256,
                "dropout": 0.1,
                "max_len": 5000,  # Adjust based on theta_dim + x_dim
                "tokenizer": {"name": "LinearTokenizer", "encoding_dim": 64},
                "use_output_scale_fn": True,
            },
            sde_config_dict={
                "name": "VPSDE",  # or "VESDE"
                "beta_min": 0.1,
                "beta_max": 20.0,
                "num_steps": 1000,
                "T_min": 1e-05,
                "T_max": 1.0,
            },
            train_config_dict={
                "learning_rate": 1e-4,  # Initial learning rate for training # used
                "min_learning_rate": 1e-6,  # Minimum learning rate for training # used
                "z_score_data": True,  # Whether to z-score the data # used
                "total_number_steps_scaling": 5,  # Scaling factor for total number of steps
                "max_number_steps": 1e9,  # Maximum number of steps for training # used
                "min_number_steps": 1e5,  # Minimum number of steps for training # used
                "training_batch_size": 64,  # Batch size for training # used
                "val_every": 100,  # Validate every 100 steps # used
                "clip_max_norm": 10.0,  # Gradient clipping max norm # used
                "condition_mask_fn": {
                    "name": "joint"
                },  # Use the base mask function defined in the task
                "edge_mask_fn": {"name": "none"},
                "validation_fraction": 0.1,  # Fraction of data to use for validation # used
                "val_repeat": 5,  # Number of times to repeat validation # used
                "stop_early_count": 5,  # Number of steps to wait before stopping early # used
                "rebalance_loss": False,  # Whether to rebalance the loss # used
            },
        )

    empirical_model_fitter.run_single_sbi(**args)

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
