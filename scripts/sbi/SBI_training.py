import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="SBI SED Fitting")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import altair as alt
    import pandas as pd
    import sbi
    import torch
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from ltu_ili_testing import SBI_Fitter, create_uncertainity_models_from_EPOCHS_cat
    from typing import Dict, List
    import numpy as np
    from simple_parsing import ArgumentParser
    from dataclasses import dataclass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alt.data_transformers.enable("vegafusion")


@app.cell(hide_code=True)
def _():
    # Setup parsing
    parser = ArgumentParser(description="SBI SED Fitting")


    @dataclass
    class Args:
        learning_rate: float = 1e-4
        stop_after_epochs: int = 20
        training_batch_size: int = 64
        validation_fraction: float = 0.1
        clip_max_norm: float = 5.0
        backend: str = "sbi"
        engine: str = "NPE"
        n_nets: int = 1
        grid: str = ""
        name: str = "sbi_sed_fitting"


    parser.add_arguments(Args, dest="args")


    def parse_args():
        if mo.running_in_notebook():
            return None, None
        else:
            args = parser.parse_args()
            return args.foo, args.options


    args, options = parse_args()
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""### Running SBI on device:{device} with torch {torch.__version__}""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Firstly we setup empirical noise models to estimate flux uncertainity from real data which we can use to scatter the photometry. There are quite a few subtlites in how these flux uncertanties are estimated and scattered, and how upper limits are handled."""
    )
    return


@app.cell(hide_code=True)
def _():
    file = "/home/tharvey/Downloads/JADES-Deep-GS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV.fits"
    table = Table.read(file)
    bands = [i.split("_")[-1] for i in table.colnames if i.startswith("loc_depth")]  # remove F606W
    new_band_names = ["HST/ACS_WFC.F606W"] + [f"JWST/NIRCam.{band.upper()}" for band in bands[1:]]

    names = [name for name in table.colnames if len(table[name].shape) <= 1]
    # df = table[names].to_pandas()

    empirical_noise_models = create_uncertainity_models_from_EPOCHS_cat(
        file, bands[1:], new_band_names[1:], plot=False
    )
    return bands, empirical_noise_models, table


@app.cell(hide_code=True)
def _(bands):
    mo.md(rf"""Created noise models for {", ".join(bands)} ({len(bands)} filters)""")
    return


@app.cell(hide_code=True)
def _(empirical_noise_models):
    model_name = "BPASS_Chab_DelayedExpSFH_0.01_z_12_CF00_v1"
    grid_path = "/home/tharvey/work/output/grid_BPASS_DelayedExponential_SFH_0.01_z_12_logN_5.7_Chab_CF00_v1.hdf5"

    empirical_model_fitter = SBI_Fitter.init_from_hdf5(model_name, grid_path, return_output=False)

    unused_filters = [
        filt
        for filt in empirical_model_fitter.raw_photometry_names
        if filt not in list(empirical_noise_models.keys())
    ]

    len(empirical_model_fitter.parameter_array)
    return empirical_model_fitter, grid_path, model_name, unused_filters


@app.cell(hide_code=True)
def _(grid_path, model_name):
    mo.md(
        rf"""Created SBI Fitter with name: {model_name} from Synthesizer model library {grid_path.split("/")[-1]}"""
    )
    return


@app.cell(hide_code=True)
def _(empirical_model_fitter, empirical_noise_models, unused_filters):
    # mo.stop(not run_button.value, mo.md("Click 👆 to create feature array."))
    with mo.redirect_stdout() and mo.redirect_stderr():
        arr = empirical_model_fitter.create_feature_array_from_raw_photometry(
            extra_features=[],
            normalize_method=None,
            include_errors_in_feature_array=True,
            scatter_fluxes=10,
            empirical_noise_models=empirical_noise_models,
            photometry_to_remove=unused_filters,
            norm_mag_limit=40,
            drop_dropouts=True,
            drop_dropout_fraction=0.4,
        )
    return (arr,)


@app.cell(hide_code=True)
def _(arr, empirical_model_fitter):
    print(len(empirical_model_fitter.parameter_array))
    _ = arr
    if empirical_model_fitter.feature_array is not None:
        fitter_df = empirical_model_fitter.create_dataframe()
        mo.output.append(fitter_df)

    print(len(empirical_model_fitter.parameter_array))
    return (fitter_df,)


@app.cell(hide_code=True)
def _(empirical_model_fitter, fitter_df):
    def create_histogram_chart(
        _df: pd.DataFrame, column: str, width: int = 240, height: int = 100
    ) -> alt.Chart:
        """
        Creates an Altair histogram chart with a prominent title and formatted x-axis labels.

        Args:
            _df: The pandas DataFrame containing the data.
            column: The name of the column to plot.
            width: The width of the chart.
            height: The height of the chart.

        Returns:
            An Altair Chart object.
        """
        chart = (
            alt.Chart(_df)
            .mark_bar(binSpacing=1)
            .encode(
                x=alt.X(
                    field=column,
                    type="quantitative",
                    bin=alt.Bin(maxbins=40),  # Adjust number of bins for clarity
                    axis=alt.Axis(
                        title=column,
                        labels=True,
                        labelAngle=0,  # Ensure labels are horizontal
                        format="~s",  # Use SI-prefix format for numbers (e.g., 1.5k, 2M)
                        # You can also use 'd' for integers or '.2f' for floats.
                    ),
                ),
                y=alt.Y(
                    "count()",
                    type="quantitative",
                    title="Number of records",
                ),
                tooltip=[
                    alt.Tooltip(
                        column,
                        type="quantitative",
                        bin=True,
                        title=column,
                        format=",.2f",
                    ),
                    alt.Tooltip(
                        "count()",
                        type="quantitative",
                        format=",.0f",
                        title="Number of records",
                    ),
                ],
            )
            .properties(
                title=alt.TitleParams(
                    text=f"Distribution of {column}",  # More descriptive title
                    anchor="middle",  # Center the title
                ),
                width=width,
                height=height,
            )
        )
        return chart


    _col = empirical_model_fitter.feature_names
    _ncharts = len(_col)
    _nrows = int(np.ceil(_ncharts / 3))

    _k = 1

    rows = []
    for _i in range(_nrows):
        _row = mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md(empirical_model_fitter.feature_names[_k + _l]),
                        create_histogram_chart(
                            fitter_df, empirical_model_fitter.feature_names[_k + _l].replace(".", "\.")
                        ),
                    ]
                )
                for _l in range(3)
            ]
        )

        rows.append(_row)
        _k += 2


    mo.vstack(rows)
    return


@app.cell(hide_code=True)
def _(arr, empirical_model_fitter):
    _ = arr
    flux_limit_slider = mo.ui.slider(
        label="Mag Limit",
        value=32,
        start=np.nanmin(empirical_model_fitter.feature_array[:, :9]),
        stop=np.nanmax(empirical_model_fitter.feature_array[:, :9]),
    )
    mo.hstack([flux_limit_slider])
    return (flux_limit_slider,)


@app.cell(hide_code=True)
def _(empirical_model_fitter, flux_limit_slider):
    n_below_1sig = np.sum(
        np.all(empirical_model_fitter.feature_array[:, :9] > flux_limit_slider.value, axis=1), axis=0
    )
    mo.output.append(
        mo.md(
            "Let's check roughly what fraction of our test samples are below our detection limit for this noise model."
        )
    )
    mo.output.append(
        mo.callout(
            f"{n_below_1sig} rows of feature array have flux > {flux_limit_slider.value} in all bands.",
            "danger",
        )
    )
    return


@app.cell(hide_code=True)
def _(fitter_df):
    text = mo.md("We can see what $f(\sigma)$ looks like")
    dropdown_x = mo.ui.dropdown(options=fitter_df.columns, label="x", value="JWST/NIRCam.F444W")
    dropdown_y = mo.ui.dropdown(options=fitter_df.columns, label="y", value="unc_JWST/NIRCam.F444W")

    mo.vstack([text, dropdown_x, dropdown_y])
    return dropdown_x, dropdown_y


@app.cell(hide_code=True)
def _(dropdown_x, dropdown_y, fitter_df):
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(fitter_df[dropdown_x.value], fitter_df[dropdown_y.value], s=1)
    ax.set_xlabel(dropdown_x.value)
    ax.set_ylabel(dropdown_y.value)
    return (fig,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Now we can setup and run the SBI model itself.""")
    return


@app.cell(hide_code=True)
def _():
    _model_options_list: Dict[str, List[str]] = {
        "sbi": ["mdn", "maf", "nsf", "made", "linear", "mlp", "resnet"],
        "lampe": ["mdn", "maf", "nsf", "ncsf", "cnf", "nice", "sospf", "gf", "naf"],
    }
    model_options: Dict[str, Dict[str, str]] = {
        backend: {model: model for model in models} for backend, models in _model_options_list.items()
    }
    # --- Cell 4: Pre-instantiate all UI components ---
    # To preserve state when n_nets changes, we create all possible UI elements
    # upfront and store them. Marimo will only display the active ones.


    # --- Cell 4: Pre-instantiate all UI components (Corrected for Independence) ---

    # FIX: To ensure each network has independent controls, the UI elements
    # (mo.ui.slider, mo.ui.dropdown) must be created *inside* the loop.
    # This creates new, distinct objects for each network's control dictionary.

    MAX_NETS = 6
    network_controls = []


    for _ in range(MAX_NETS):
        # Each iteration creates brand new UI objects
        network_controls.append(
            {
                "model_type": mo.ui.dropdown(
                    options=model_options["sbi"],
                    value="mdn",
                    label="Model Type",
                ),
                "hidden_features": mo.ui.slider(
                    start=10,
                    stop=256,
                    step=1,
                    value=50,
                    label="Hidden Features:",
                    show_value=True,
                ),
                "value_slider": mo.ui.slider(
                    start=1,
                    stop=20,
                    step=1,
                    value=4,
                    label="Num Transforms/Components:",
                    show_value=True,
                ),
            }
        )


    # Create the initial state for all possible networks
    initial_configs = [
        {
            "model_type": "mdn",
            "hidden_features": 50,
            "value": 4,  # Represents num_components or num_transforms
        }
        for _ in range(MAX_NETS)
    ]

    # Initialize the state and get the setter function
    configs, set_configs = mo.state(initial_configs, allow_self_loops=False)

    # Initialize the state and get the setter function
    # These handler functions are called by the UI elements' on_change event.


    def update_config_value(key: str, value, index: int) -> None:
        """A generic handler to update any key for a given network."""
        # It's good practice to work on a copy of the state
        current_configs = list(configs())
        current_configs[index][key] = value
        set_configs(current_configs)
    return configs, model_options, network_controls, update_config_value


@app.cell(hide_code=True)
def _(
    backend,
    configs,
    model_options: Dict[str, Dict[str, str]],
    n_nets,
    update_config_value,
):
    # --- Cell 5: Dynamically build and display the network configuration UI (Corrected for Reactivity) ---

    # FIX: All logic for updating and displaying the tabs MUST be in the same
    # cell. This allows Marimo to correctly track that this cell needs to
    # re-run when `backend` or any of the `network_controls` UI elements change.

    mo.md("---")
    mo.md("## Network Ensemble Configuration")


    # First, update the state of all dropdowns based on the global backend
    _backend_value = backend.value
    num_networks: int = n_nets.value
    tabs = {}

    # Get the current backend to populate dropdown options
    available_models = model_options[backend.value]

    for _i in range(num_networks):
        # Get the config for this specific network from the central state
        net_config = configs()[_i]

        # Create UI elements on-the-fly based on the current state
        model_type_dd = mo.ui.dropdown(
            options=available_models,
            value=net_config["model_type"],
            label="Model Type",
            # When this dropdown changes, call the handler for the correct index `i`
            on_change=lambda value, index=_i: update_config_value("model_type", value, index),
        )

        hidden_features_slider = mo.ui.slider(
            start=10,
            stop=600,
            value=net_config["hidden_features"],
            label="Hidden Features:",
            include_input=True,
            on_change=lambda value, index=_i: update_config_value("hidden_features", value, index),
        )

        value_slider = mo.ui.slider(
            start=1,
            stop=20,
            value=net_config["value"],
            include_input=True,
            on_change=lambda value, index=_i: update_config_value("value", value, index),
            label="Num Components/Transforms:",
        )

        # Build the list of elements to show in the tab
        current_model: str = net_config["model_type"]
        ui_elements = [model_type_dd]

        if current_model == "mdn":
            value_slider.label = "Num Components"
            ui_elements.extend([hidden_features_slider, value_slider])
        elif current_model in ["maf", "nsf", "made", "ncsf", "cnf", "gf", "sospf", "naf", "unaf"]:
            value_slider.label = "Num Transforms"
            ui_elements.extend([hidden_features_slider, value_slider])
        elif current_model in ["mlp", "resnet"]:
            ui_elements.append(hidden_features_slider)

        tabs[f"Network {_i + 1}"] = mo.vstack(ui_elements)


    """    for i in range(MAX_NETS):
    controls = network_controls[i]
    model_type_dd = controls["model_type"]
    available_models = model_options[_backend_value]
    model_type_dd.options = available_models
    # If current value is no longer valid, reset it
    if model_type_dd.value not in available_models:
        model_type_dd.value = list(available_models.keys())[0]

    # Second, build the UI from the now-updated controls
    tabs = {}
    num_networks: int = n_nets.value

    for i in range(num_networks):
        controls = network_controls[i]
        model_type_dd = controls["model_type"]
        hidden_features_slider = controls["hidden_features"]
        value_slider = controls["value_slider"]

        current_model: str = model_type_dd.value
        ui_elements: List[mo.ui.UIElement[Any, Any]] = [model_type_dd]

        if current_model == "mdn":
            value_slider.label = "Num Components"
            ui_elements.extend([hidden_features_slider, value_slider])
        elif current_model in ["maf", "nsf", "made", "ncsf", "cnf", "gf", "sospf", "naf", "unaf"]:
            value_slider.label = "Num Transforms"
            ui_elements.extend([hidden_features_slider, value_slider])
        elif current_model in ["mlp", "resnet"]:
            ui_elements.append(hidden_features_slider)

        tabs[f"Network {i + 1}"] = mo.vstack(ui_elements)""";


    return (tabs,)


@app.cell(hide_code=True)
def _():
    mo.output.append(mo.md("###  Base Setup"))

    unique_name = mo.ui.text(placeholder="unique name for run", label="Name:")

    n_nets = mo.ui.slider(start=1, stop=6, value=1, label="Number of Networks:", show_value=True)
    backend = mo.ui.dropdown(options=["lampe", "sbi"], value="sbi", label="Backend:")
    engine = mo.ui.dropdown(
        options=["NPE", "NLE", "NRE", "SNPE", "SNLE", "SNRE"], value="NPE", label="Engine:"
    )
    stop_after_epochs = mo.ui.slider(
        start=5, stop=50, value=20, label="Stop after Epochs: ", show_value=True
    )

    steps = np.logspace(-2, -6, 100)
    learning_rate = mo.ui.slider(
        steps=steps, label="Learning Rate:", show_value=True, value=steps[len(steps) // 2]
    )
    validation_fraction = mo.ui.slider(
        start=0, stop=0.30, value=0.10, label="Validation Fraction:", include_input=True
    )
    clip_max_norm = mo.ui.slider(
        start=0.1, stop=5.0, value=5, label="Clip Max Norm:", include_input=True
    )
    training_batch_size = mo.ui.slider(
        start=32, stop=128, value=64, label="Training Batch Size: ", show_value=True
    )

    ui = mo.vstack(
        [
            unique_name.center(),
            mo.hstack(
                [
                    mo.vstack([n_nets, backend, engine, learning_rate], align="stretch"),
                    mo.vstack(
                        [stop_after_epochs, validation_fraction, clip_max_norm, training_batch_size]
                    ),
                ]
            ),
        ]
    )

    mo.output.append(ui)
    return (
        backend,
        clip_max_norm,
        engine,
        learning_rate,
        n_nets,
        stop_after_epochs,
        training_batch_size,
        unique_name,
        validation_fraction,
    )


@app.cell(hide_code=True)
def _(tabs):
    mo.output.append(mo.md("### Network Configuration"))
    mo.output.append(mo.ui.tabs(tabs))
    return


@app.cell(hide_code=True)
def _(
    backend,
    clip_max_norm,
    configs,
    engine,
    learning_rate,
    n_nets,
    network_controls,
    stop_after_epochs,
    training_batch_size,
    unique_name,
    validation_fraction,
):
    n_nets.value, backend.value, engine.value, stop_after_epochs.value, learning_rate.value


    def get_configs() -> Dict:
        """Collects and returns the current configuration values."""
        config: Dict = {
            "n_nets": n_nets.value,
            "backend": backend.value,
            "engine": engine.value,
            "learning_rate": np.round(learning_rate.value, 7),
            "stop_after_epochs": stop_after_epochs.value,
            "training_batch_size": training_batch_size.value,
            "clip_max_norm": clip_max_norm.value,
            "validation_fraction": validation_fraction.value,
            "model_type": [],
            "hidden_features": [],
            "num_components": [],
            "num_transforms": [],
        }
        for _i in range(n_nets.value):
            controls = network_controls[_i]
            model_type = controls["model_type"].value
            config["model_type"].append(model_type)

            value_slider_val = controls["value_slider"].value
            hidden_features_val = controls["hidden_features"].value

            if model_type == "mdn":
                config["hidden_features"].append(hidden_features_val)
                config["num_components"].append(value_slider_val)
            elif model_type in ["maf", "nsf", "made", "ncsf", "cnf", "gf", "sospf", "naf", "unaf"]:
                config["hidden_features"].append(hidden_features_val)
                config["num_transforms"].append(value_slider_val)
            elif model_type in ["mlp", "resnet"]:
                config["hidden_features"].append(hidden_features_val)

        # Clean up empty lists to make the table tidy
        final_config = {k: v for k, v in config.items() if v}
        return final_config


    num_nets_to_show = n_nets.value
    current_configs = configs()[:num_nets_to_show]

    # Format the raw config data into the desired output dictionary
    output_dict = {
        "name": unique_name.value,
        "n_nets": n_nets.value,
        "backend": backend.value,
        "engine": engine.value,
        "learning_rate": np.round(learning_rate.value, 7),
        "stop_after_epochs": stop_after_epochs.value,
        "training_batch_size": training_batch_size.value,
        "clip_max_norm": clip_max_norm.value,
        "validation_fraction": validation_fraction.value,
        "model_type": [net["model_type"] for net in current_configs],
        "hidden_features": [],
        "num_components": [],
        "num_transforms": [],
    }

    for n_config in current_configs:
        model_type = n_config["model_type"]
        if model_type in ["mdn", "mlp", "resnet"] or "maf" in model_type or "nsf" in model_type:
            output_dict["hidden_features"].append(n_config["hidden_features"])

        if model_type == "mdn":
            output_dict["num_components"].append(n_config["value"])
        elif "maf" in model_type or "nsf" in model_type or "made" in model_type:
            output_dict["num_transforms"].append(n_config["value"])

    # Clean up empty lists to make the table tidy
    config = {k: v for k, v in output_dict.items() if v}

    show_config = {
        k: ", ".join([str(_w) for _w in v]) if isinstance(v, list) else v for k, v in config.items()
    }


    collected_model_types = config["model_type"]
    collected_hidden_features = config["hidden_features"]
    collected_num_transforms = config["num_transforms"] if "num_transforms" in config.keys() else [-1]
    collected_num_components = config["num_components"] if "num_components" in config.keys() else [-1]

    collected_model_types = (
        collected_model_types[0] if len(collected_model_types) == 1 else collected_model_types
    )
    collected_hidden_features = (
        collected_hidden_features[0] if len(collected_model_types) == 1 else collected_hidden_features
    )
    collected_num_transforms = (
        collected_num_transforms[0] if len(collected_num_transforms) == 1 else collected_num_transforms
    )
    collected_num_components = (
        collected_num_components[0] if len(collected_num_components) == 1 else collected_num_components
    )

    mo.output.append(mo.ui.table(show_config, pagination=False))
    return (
        collected_hidden_features,
        collected_model_types,
        collected_num_components,
        collected_num_transforms,
    )


@app.cell(hide_code=True)
def _():
    run_sbi_button = mo.ui.run_button(label="Run SBI Training")
    run_sbi_button
    return (run_sbi_button,)


@app.cell(hide_code=True)
def _(
    backend,
    clip_max_norm,
    collected_hidden_features,
    collected_model_types,
    collected_num_components,
    collected_num_transforms,
    empirical_model_fitter,
    engine,
    learning_rate,
    n_nets,
    run_sbi_button,
    stop_after_epochs,
    training_batch_size,
    unique_name,
    validation_fraction,
):
    mo.stop(not run_sbi_button.value, mo.md("Click 👆 to run SBI training."))

    empirical_model_fitter.run_single_sbi(
        n_nets=n_nets.value,
        backend=backend.value,
        engine=engine.value,
        stop_after_epochs=stop_after_epochs.value,
        learning_rate=learning_rate.value,
        hidden_features=collected_hidden_features,
        num_transforms=collected_num_transforms,
        num_components=collected_num_components,
        model_type=collected_model_types,
        training_batch_size=training_batch_size.value,
        validation_fraction=validation_fraction.value,
        clip_max_norm=clip_max_norm.value,
        name_append=unique_name.value if unique_name.value else "timestamp",
    )
    return


@app.cell
def _(empirical_model_fitter):
    d = empirical_model_fitter.load_model_from_pkl(
        "/home/tharvey/work/ltu-ili_testing/models/BPASS_Chab_DelayedExpSFH_0.01_z_12_CF00_v1/BPASS_Chab_DelayedExpSFH_0.01_z_12_CF00_v1noise_large_sbi_mdn_posterior.pkl"
    )
    d = d[2]
    print(np.shape(d["test_indices"]), np.shape(d["train_indices"]), np.shape(d["feature_array"]))
    return


@app.cell
def _(empirical_model_fitter):
    empirical_model_fitter.plot_diagnostics(
        plots_dir=f"/home/tharvey/work/ltu-ili_testing/models/name/plots/no_noise/"
    )
    return


@app.cell
def _(empirical_model_fitter, fig, table):
    selected_cat = table[table["final_sample_highz_fsps_larson_no_bd"] == True]

    selected_cat["NUMBER", "MAG_APER_f444W_aper_corr", "zbest_fsps_larson"]
    import astropy.units as u

    plot = True
    all_samples = []
    X_tests = []

    jades_filters = ["f090W", "f115W", "f150W", "f200W", "f277W", "f335M", "f356W", "f410M", "f444W"]
    for id in selected_cat["NUMBER"]:
        row = selected_cat[selected_cat["NUMBER"] == id]

        flux = []
        err = []

        for filter_name in jades_filters:
            f = row[f"MAG_APER_{filter_name}_aper_corr"][0][0]
            if f == 99:
                f = 50

            e = (row[f"loc_depth_{filter_name}"][0, 0] * u.ABmag).to("Jy").value / 5
            fi = row[f"FLUX_APER_{filter_name}_aper_corr_Jy"][0][0]
            snr = fi / e

            if snr < 1 or not np.isfinite(snr):
                f = 50.0
                e = 1.084
            else:
                e = np.abs(2.5 * e / (np.log(10) * fi))

            flux.append(f)
            err.append(e)

        flux = np.array(flux)
        err = np.array(err)

        x_test = np.concatenate((flux, err))
        X_tests.append(x_test)

    all_samples = empirical_model_fitter.sample_posterior(X_test=X_tests)


    # plot posterior sample histogram
    if plot:
        for j, id in enumerate(selected_cat["NUMBER"][:10]):
            figi, axs = plt.subplots(
                2,
                int(np.ceil((len(empirical_model_fitter.fitted_parameter_names) / 2))),
                figsize=(20, 6),
                sharey=True,
            )
            axs = axs.flatten()
            for i, name in enumerate(empirical_model_fitter.simple_fitted_parameter_names):
                axs[i].hist(
                    all_samples[j, :, i], bins=50, density=True, alpha=0.5, label="Posterior Samples"
                )
                axs[i].set_title(name)
                axs[i].set_xlabel(name)
                axs[i].set_ylabel("Density")
                axs[i].legend()

            true_redshift = row["zbest_fsps_larson"][0]
            axs[0].axvline(true_redshift, color="red", linestyle="--", label="True Redshift")
            fig.suptitle(f"Posterior Samples for ID {id}", fontsize=16)
            plt.tight_layout()
            mo.output.append(figi)


    all_samples = np.array(all_samples)

    all_redshifts = all_samples[:, :, 0]

    print(all_redshifts.shape)
    return all_redshifts, selected_cat


@app.cell
def _(all_redshifts, selected_cat):
    upper, median, lower = np.percentile(all_redshifts, [84, 50, 16], axis=1)

    all_table_redshifts = np.squeeze(selected_cat["zbest_fsps_larson"].data)
    all_table_upper = np.squeeze(selected_cat["zbest_16_fsps_larson"].data)
    all_table_lower = np.squeeze(selected_cat["zbest_84_fsps_larson"].data)
    mask = selected_cat["zbest_fsps_larson"].data < 12
    all_table_redshifts = all_table_redshifts[mask]
    all_table_upper = all_table_upper[mask]
    all_table_lower = all_table_lower[mask]
    upper = upper[mask]
    median = median[mask]
    lower = lower[mask]

    plt.figure(figsize=(10, 6))

    xerr = np.abs(
        np.array([all_table_redshifts - all_table_lower, all_table_upper - all_table_redshifts])
    )
    yerr = np.abs(np.array([lower - median, upper - median]))


    plt.errorbar(
        all_table_redshifts,
        median,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        label="Empirical Model",
        color="blue",
        alpha=0.5,
    )

    # add a 1:1
    plt.plot([5, 12], [5, 12], color="white", linestyle="--", label="1:1 Line")

    plt.xlabel("EAZY Redshift")
    plt.ylabel("SBI Model Redshift")
    return


@app.cell
def _(empirical_model_fitter):
    len(empirical_model_fitter.fitted_parameter_names)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
