import os
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
)  # For GalaxySimulator type hints

import corner
import jax
import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import OmegaConf  # To create DictConfig-like objects if needed
from scoresbibm.methods.score_transformer import train_transformer_model
from scoresbibm.tasks.base_task import InferenceTask
from synthesizer.emission_models import (
    TotalEmission,
)
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.parametric import (
    SFH,
    ZDist,
)  # Need concrete SFH, ZDist classes
from unyt import (
    Myr,
)

from sbifitter import GalaxySimulator

file_path = os.path.dirname(os.path.realpath(__file__))
grid_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "grids")
output_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Class for Prior ---
class GalaxyPrior:
    """
    A simple prior distribution handler for galaxy parameters.
    Samples uniformly from ranges specified for each parameter.
    """

    def __init__(
        self,
        prior_ranges: Dict[str, Tuple[float, float]],
        param_order: List[str],
    ):
        self.prior_ranges = prior_ranges
        self.param_order = param_order
        self.theta_dim = len(param_order)

        self.distributions = []
        for param_name in self.param_order:
            low, high = self.prior_ranges[param_name]
            self.distributions.append(
                torch.distributions.Uniform(
                    torch.tensor(float(low)), torch.tensor(float(high))
                )
            )

    def sample(
        self, sample_shape: Tuple[int], sample_lhc=False, rng=None
    ) -> torch.Tensor:
        """
        Generates samples from the prior.
        Args:
            sample_shape: A tuple containing the number of samples, e.g., (num_samples,).
        Returns:
            A PyTorch tensor of shape (num_samples, theta_dim).
        """
        if not sample_lhc:
            num_samples = sample_shape[0]
            samples_per_param = [
                dist.sample((num_samples, 1)) for dist in self.distributions
            ]
        else:
            # Use boundaries, but sample from Latin Hypercube
            from scipy.stats.qmc import LatinHypercube

            sampler = LatinHypercube(d=self.theta_dim, rng=rng)
            lhc_samples = sampler.random(n=sample_shape[0])
            samples_per_param = []
            for i, dist in enumerate(self.distributions):
                low, high = self.prior_ranges[self.param_order[i]]
                # Scale LHC samples to the range of the distribution
                scaled_samples = low + (high - low) * lhc_samples[:, i : i + 1]
                samples_per_param.append(
                    torch.tensor(scaled_samples, dtype=torch.float32)
                )

        return torch.cat(samples_per_param, dim=1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of theta under the prior.
        Assumes theta is a batch of shape (num_samples, theta_dim).
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # Make it (1, theta_dim)

        log_probs_per_param = []
        for i, dist in enumerate(self.distributions):
            log_probs_per_param.append(dist.log_prob(theta[:, i]))

        # Sum log_probs for independent parameters
        return torch.sum(torch.stack(log_probs_per_param, dim=1), dim=1)


# --- Custom Simformer Task ---
class GalaxyPhotometryTask(InferenceTask):
    """
    A Simformer InferenceTask for the GalaxySimulator.
    """

    _theta_dim: int
    _x_dim: int

    def __init__(
        self,
        name: str = "galaxy_photometry_task",
        backend: str = "jax",
        prior_dict: Dict[str, Tuple[float, float]] = None,
        param_names_ordered: List[str] = None,
        run_simulator_fn: Callable = None,
        num_filters: int = None,
    ):
        super().__init__(name, backend)

        if (
            prior_dict is None
            or param_names_ordered is None
            or run_simulator_fn is None
            or num_filters is None
        ):
            raise ValueError(
                """prior_dict, param_names_ordered, run_simulator_fn,
                and num_filters must be provided."""
            )

        self.param_names_ordered = param_names_ordered
        self._theta_dim = len(param_names_ordered)
        self._x_dim = num_filters

        self.prior_dist = GalaxyPrior(
            prior_ranges=prior_dict, param_order=self.param_names_ordered
        )
        self.run_simulator_fn = run_simulator_fn

    def get_theta_dim(self) -> int:
        return self._theta_dim

    def get_x_dim(self) -> int:
        return self._x_dim

    def get_prior(self):
        """Returns the prior distribution object."""
        return self.prior_dist

    def get_simulator(self):
        """
        Returns a callable that takes a batch of thetas and returns a batch of xs.
        The provided run_simulator_fn processes one sample at a time
        and returns a torch tensor.
        This wrapper will handle batching.
        """

        def batched_simulator(
            thetas_batch_torch: torch.Tensor,
        ) -> torch.Tensor:
            xs_list = []
            for i in range(thetas_batch_torch.shape[0]):
                theta_sample_torch = thetas_batch_torch[i, :]
                # run_simulator_fn expects numpy array if it's not a dict,
                # and handles tensor conversion internally.
                # It returns a tensor of shape [1, num_filters].
                x_sample_torch = self.run_simulator_fn(
                    theta_sample_torch, return_type="tensor"
                )
                xs_list.append(x_sample_torch)
            return torch.cat(xs_list, dim=0)  # Shape will be (num_samples, num_filters)

        return batched_simulator

    def get_data(self, num_samples: int, **kwargs) -> Dict[str, jnp.ndarray]:
        """Generates and returns a dictionary of {'theta': thetas, 'x': xs} as JAX arr"""
        prior = self.get_prior()
        simulator = self.get_simulator()  # This is our batched_simulator

        # Sample thetas (parameters) using the prior
        # GalaxyPrior.sample returns a PyTorch tensor
        thetas_torch = prior.sample((num_samples,), **kwargs)

        # Simulate xs (photometry) using the parameters
        # batched_simulator also returns a PyTorch tensor
        xs_torch = simulator(thetas_torch)

        if self.backend == "jax":
            thetas_out = jnp.array(thetas_torch.cpu().numpy())
            xs_out = jnp.array(xs_torch.cpu().numpy())
        elif self.backend == "numpy":
            thetas_out = thetas_torch.cpu().numpy()
            xs_out = xs_torch.cpu().numpy()
        else:  # "torch" or other
            thetas_out = thetas_torch
            xs_out = xs_torch

        return {"theta": thetas_out, "x": xs_out}

    def get_node_id(self) -> jnp.ndarray:
        """Returns an array identifying the nodes (dimensions) of theta and x."""
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":  # Should align with SBIBMTask if that's a reference
            return torch.arange(dim)
        else:  # JAX or numpy
            return jnp.arange(dim)

    def get_base_mask_fn(self) -> Callable:
        """Defines the base attention mask for the transformer."""
        theta_dim = self.get_theta_dim()
        x_dim = self.get_x_dim()

        # Parameters only attend to themselves (or causal if ordered)
        thetas_self_mask = jnp.eye(theta_dim, dtype=jnp.bool_)

        # Data can attend to previous/current data points (causal within x)
        # Or use jnp.ones if full self-attention within x is desired.
        xs_self_mask = jnp.tril(jnp.ones((x_dim, x_dim), dtype=jnp.bool_))

        # Data can attend to all parameters
        xs_attend_thetas_mask = jnp.ones((x_dim, theta_dim), dtype=jnp.bool_)

        # Parameters do not attend to data
        thetas_attend_xs_mask = jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)

        base_mask = jnp.block(
            [
                [thetas_self_mask, thetas_attend_xs_mask],
                [xs_attend_thetas_mask, xs_self_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            # Handles potential permutation/subsetting of nodes
            return base_mask[jnp.ix_(node_ids, node_ids)]

        return base_mask_fn


if __name__ == "__main__":
    # Define sfh and zdist instances or classes as used by GalaxySimulator
    sfh_model_class = SFH.LogNormal
    zdist_model_class = ZDist.DeltaConstant

    # Example: Define global 'device' if not already defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_dir = os.environ["SYNTHESIZER_GRID_DIR"]

    # path for this file

    dir_path = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(dir_path)), "grids/")

    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"

    # --- This part needs functional grid, instrument etc. ---

    grid = Grid(grid_name, grid_dir=grid_dir)
    filter_codes = [
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F444W",
    ]
    filterset = FilterCollection(filter_codes)
    instrument = Instrument("JWST", filters=filterset)
    emission_model_instance = TotalEmission(
        grid=grid,
        fesc=0.0,
        fesc_ly_alpha=0.1,
        dust_curve=Calzetti2000(),
        dust_emission_model=None,
    )
    emitter_params_dict = {"stellar": ["tau_v"]}
    galaxy_simulator_instance = GalaxySimulator(
        sfh_model=sfh_model_class,  # Pass the class
        zdist_model=zdist_model_class,  # Pass the class
        grid=grid,
        instrument=instrument,
        emission_model=emission_model_instance,
        emission_model_key="total",
        emitter_params=emitter_params_dict,
        param_units={"peak_age": Myr, "max_age": Myr},  # Ensure Myr is defined
        normalize_method=None,
        output_type="photo_fnu",
        out_flux_unit="ABmag",
    )

    inputs_list = [
        "redshift",
        "log_mass",
        "log10metallicity",
        "tau_v",
        "peak_age",
        "max_age",
        "tau",
    ]
    filter_codes_list = [
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F444W",
    ]

    priors_ranges_dict = {
        "redshift": (5.0, 10.0),
        "log_mass": (7.0, 11.0),
        "log10metallicity": (-3.0, -1.3),
        "tau_v": (0.0, 2),
        "peak_age": (
            0.0,
            500.0,
        ),  # Ensure float for peak_age if used with torch.tensor(float(low))
        "max_age": (500.0, 1000.0),
        "tau": (0.3, 1.5),
    }

    def run_simulator_glob(params, return_type="tensor"):
        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        if isinstance(params, dict):
            pass  # assumes params are correctly keyed
        elif isinstance(params, (list, tuple, np.ndarray)):
            params = np.squeeze(params)
            params = {inputs_list[i]: params[i] for i in range(len(inputs_list))}

        phot = galaxy_simulator_instance(
            params
        )  # This line requires galaxy_simulator_instance

        if return_type == "tensor":
            return torch.tensor(phot[np.newaxis, :], dtype=torch.float32).to(device)
        else:
            return phot

    galaxy_task = GalaxyPhotometryTask(
        prior_dict=priors_ranges_dict,
        param_names_ordered=inputs_list,
        run_simulator_fn=run_simulator_glob,
        num_filters=len(filter_codes_list),
    )

    # Test data generation
    print(f"Theta dim: {galaxy_task.get_theta_dim()}")
    print(f"X dim: {galaxy_task.get_x_dim()}")
    data_batch = galaxy_task.get_data(num_samples=3)
    print("Sampled theta (JAX):", data_batch["theta"])
    print("Shape of theta:", data_batch["theta"].shape)
    print("Sampled x (JAX):", data_batch["x"])
    print("Shape of x:", data_batch["x"].shape)

    # Test prior sampling directly
    prior_for_test = galaxy_task.get_prior()
    theta_samples_torch = prior_for_test.sample((2,))
    print("Direct prior samples (Torch):", theta_samples_torch)
    print(
        "Log prob of prior samples:",
        prior_for_test.log_prob(theta_samples_torch),
    )

    # Test base mask function
    mask_fn = galaxy_task.get_base_mask_fn()
    node_ids_example = jnp.arange(galaxy_task.get_theta_dim() + galaxy_task.get_x_dim())
    applied_mask = mask_fn(node_ids=node_ids_example, node_meta_data=None)
    print("Base mask applied to node_ids:", applied_mask)

    # Model Configuration (from config/method/model/score_transformer.yaml)
    #
    model_config_dict = {
        "name": "ScoreTransformer",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "d_feedforward": 256,
        "dropout": 0.1,
        "max_len": 5000,  # Adjust based on theta_dim + x_dim
        "tokenizer": {"name": "LinearTokenizer", "encoding_dim": 64},
        "use_output_scale_fn": True,
        # Add other model-specific parameters as per the YAML
    }

    # SDE Configuration (e.g., from config/method/sde/vpsde.yaml)
    #
    sde_config_dict = {
        "name": "VPSDE",  # or "VESDE"
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_steps": 1000,
        "T_min": 1e-05,
        "T_max": 1.0,
        # "likelihood_weighting": False,
        # "schedule_name": "linear",
        # Add other SDE-specific parameters
    }

    # Training Configuration (from config/method/train/train_score_transformer.yaml)
    #
    train_config_dict = {
        "learning_rate": 1e-4,  # Initial learning rate for training # used
        "min_learning_rate": 1e-6,  # Minimum learning rate for training # used
        "z_score_data": True,  # Whether to z-score the data # used
        "total_number_steps_scaling": 5,  # Scaling factor for total number of steps
        "max_number_steps": 1e8,  # Maximum number of steps for training # used
        "min_number_steps": 1e4,  # Minimum number of steps for training # used
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
    }

    method_config_dict = {
        "device": str(device),  # Ensure this matches device setup
        "sde": sde_config_dict,
        "model": model_config_dict,
        "train": train_config_dict,
    }

    # Convert the main method_cfg to OmegaConf DictConfig
    method_cfg = OmegaConf.create(method_config_dict)

    print("Instantiating GalaxyPhotometryTask...")
    galaxy_task = GalaxyPhotometryTask(
        prior_dict=priors_ranges_dict,
        param_names_ordered=inputs_list,
        run_simulator_fn=run_simulator_glob,  # function to run the simulator
        num_filters=len(filter_codes_list),
        backend="jax",  # Or "torch"
    )
    print("Task instantiated.")

    # --- 2. Generate Data ---
    num_training_simulations = 5000  # Example number
    print(f"Generating {num_training_simulations} training simulations...")
    # .get_data() returns a dict with JAX arrays if backend is "jax"
    training_data = galaxy_task.get_data(num_samples=num_training_simulations)
    theta_train = training_data["theta"]
    x_train = training_data["x"]
    print(f"Data generated: theta shape {theta_train.shape}, x shape {x_train.shape}")

    # (Optional) Generate validation data if train_config.val_split is 0
    num_validation_simulations = 20
    validation_data = galaxy_task.get_data(num_samples=num_validation_simulations)
    theta_val = validation_data["theta"]
    x_val = validation_data["x"]

    # --- 3. Set RNG Seed for JAX ---
    rng_seed_for_training = 0
    master_rng_key = jax.random.PRNGKey(rng_seed_for_training)

    # --- 4. Train the Model ---

    print("Starting training...")
    trained_score_model = train_transformer_model(
        task=galaxy_task,
        data=training_data,  # Expects dict {"theta": ..., "x": ...} with JAX arrays
        method_cfg=method_cfg,  # The OmegaConf object created above
        rng=master_rng_key,
    )
    print(
        "Training finished. Model returned by train_transformer_model:",
        type(trained_score_model),
    )
    plot_corner = True
    # Take test observation
    theta_dim = galaxy_task.get_theta_dim()
    x_dim = galaxy_task.get_x_dim()
    # Mask for posterior: theta is unknown (0), x is known (1)
    posterior_condition_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    for i, xobs in enumerate(x_val):
        x_val = jnp.array([xobs], dtype=jnp.float32)
        samples = trained_score_model.sample_batched(
            num_samples=1000,
            x_o=x_val,
            rng=master_rng_key,
            condition_mask=posterior_condition_mask,
        )
        if plot_corner:
            import corner
            import matplotlib.pyplot as plt

            truth = jnp.array(theta_val[i], dtype=jnp.float32)
            corner.corner(
                samples,
                labels=galaxy_task.param_names_ordered,
                show_titles=True,
                truths=truth,
                quantiles=[0.16, 0.5, 0.84],
                title_kwargs={"fontsize": 12},
            )
            plt.savefig(f"{output_folder}/simformer/plots/corner_plot_{i}.png")

    import pickle

    with open("trained_galaxy_score_model_params.pkl", "wb") as f:
        pickle.dump(trained_score_model.score_model_params, f)
    print("Model parameters saved (example).")
