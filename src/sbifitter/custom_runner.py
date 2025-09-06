"""Custom runner like LtU-ILI's SBIRunner, but with Optuna-based hyperparam optimization."""

import logging
import time
from copy import deepcopy
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import tarp
import torch
import torch.nn as nn
import yaml
from ili.dataloaders.loaders import StaticNumpyLoader
from ili.inference import SBIRunner
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from torch.distributions import Distribution, constraints
from torch.distributions.constraint_registry import (
    _transform_to_interval,
    biject_to,
    transform_to,
)
from torch.distributions.utils import broadcast_all
from torch.optim import Adam, AdamW
from torch.types import _size
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

try:  # sbi > 0.22.0
    from sbi.inference.posteriors import EnsemblePosterior
except ImportError:  # sbi < 0.22.0
    from sbi.utils.posterior_ensemble import (
        NeuralPosteriorEnsemble as EnsemblePosterior,
    )

from ili.dataloaders import _BaseLoader
from ili.utils import load_from_config, load_nde_sbi

from . import logger
from .utils import _exit_alternate_screen, update_plot

optuna.logging.set_verbosity(optuna.logging.INFO)


class SBICustomRunner(SBIRunner):
    """Runner for SBI inference which uses a custom training loop with Optuna-based optimization."""

    def __init__(
        self,
        prior: Distribution,
        engine: str,
        net_configs: List[Dict],
        embedding_net: nn.Module = nn.Identity(),
        train_args: Dict = {},
        out_dir: Union[str, Path] = None,
        device: str = "cpu",
        proposal: Distribution = None,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = None,
    ):
        """Initialize the custom SBI runner.

        Args:
            prior (Distribution): The prior distribution for the parameters.
            engine (str): The engine to use for the inference (e.g., "NPE, NLE, SNPE").
            net_configs (List[Dict]): List of configurations for the neural networks.
            embedding_net (nn.Module, optional): The embedding network to use.
                Defaults to nn.Identity.
            train_args (Dict, optional): Training arguments including Optuna configuration.
                Defaults to an empty dictionary.
            out_dir (Union[str, Path], optional): Output directory for saving models.
                Defaults to None.
            device (str, optional): Device to use for training (e.g., "cpu", "cuda").
                Defaults to "cpu".
            proposal (Distribution, optional): The proposal distribution for the inference.
                Defaults to None.
            name (Optional[str], optional): Name for the model. Defaults to an empty string.
            signatures (Optional[List[str]], optional): List of signatures for the models.
                Defaults to None.
        """
        super().__init__(
            prior=prior,
            engine=engine,
            nets=[],
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            proposal=proposal,
            name=name,
            signatures=signatures,
        )
        self.net_configs = net_configs
        self.embedding_net = embedding_net

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "SBICustomRunner":
        """Create an instance of SBICustomRunner from a configuration file."""
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        config.update(kwargs)

        config["prior"]["args"]["device"] = config["device"]
        prior = load_from_config(config["prior"])
        proposal = None
        if "proposal" in config:
            config["proposal"]["args"]["device"] = config["device"]
            proposal = load_from_config(config["proposal"])

        embedding_net = (
            load_from_config(config["embedding_net"])
            if "embedding_net" in config
            else nn.Identity()
        )

        train_args = config["train_args"]
        out_dir = Path(config["out_dir"])
        name = config["model"].get("name", "") + "_"

        net_configs = config["model"]["nets"]
        signatures = [cfg.pop("signature", "") for cfg in net_configs]

        engine = config["model"]["engine"]

        return cls(
            prior=prior,
            proposal=proposal,
            engine=engine,
            net_configs=net_configs,
            embedding_net=embedding_net,
            device=config["device"],
            train_args=train_args,
            out_dir=out_dir,
            signatures=signatures,
            name=name,
        )

    def __call__(
        self, loader: _BaseLoader, validation_loader: Optional[_BaseLoader] = None, seed: int = None
    ) -> Tuple[Optional[EnsemblePosterior], List[Dict]]:
        """Run SBI inference.

        If `train_args['skip_optimization']` is True,
        it trains a single model with fixed parameters.
        Otherwise, it performs Optuna-based hyperparameter optimization.

        Args:
            loader (_BaseLoader): The data loader containing training data.
            validation_loader (_BaseLoader, optional): The data loader containing validation data.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            A tuple containing the trained posterior and a list of summary dictionaries.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if "optuna" in self.train_args:
                optuna.samplers.TPESampler(seed=seed)

        x_train = torch.from_numpy(loader.get_all_data()).float().to(self.device)
        theta_train = torch.from_numpy(loader.get_all_parameters()).float().to(self.device)

        final_summary = {}
        trained_estimator = None

        if self.train_args.get("skip_optimization", False):
            logger.info(
                "Skipping hyperparameter optimization."
                "Training a single model with fixed parameters."
            )

            fixed_params = self.train_args.get("fixed_params")
            if not fixed_params or "model_choice" not in fixed_params:
                raise ValueError(
                    "`skip_optimization` is True, but `fixed_params` (including 'model_choice') "
                    "are not defined in `train_args`."
                )

            study_direction = None

            trained_estimator, final_summary = self._build_and_train_model(
                params=fixed_params,
                x_train=x_train,
                theta_train=theta_train,
                study_direction=study_direction,
            )
            final_summary["best_params"] = fixed_params

        else:
            optuna_config = self.train_args["optuna"]
            x_val, theta_val = None, None

            if (
                validation_loader is None
                and "objective" in optuna_config
                and "data_config" in optuna_config["objective"]
            ):
                val_data_config_path = optuna_config["objective"]["data_config"]
                validation_loader = StaticNumpyLoader.from_config(val_data_config_path)
            if validation_loader:
                x_val = torch.from_numpy(validation_loader.get_all_data()).float().to(self.device)
                theta_val = (
                    torch.from_numpy(validation_loader.get_all_parameters()).float().to(self.device)
                )

            study_direction = optuna_config["study"].get("direction", "minimize")
            objective_fn, fixed_params_from_obj = self._create_objective(
                x_train, theta_train, x_val, theta_val, study_direction
            )

            logger.info("Starting Optuna hyperparameter search...")
            pruner_options = {
                "Hyperband": optuna.pruners.HyperbandPruner,
                "Median": optuna.pruners.MedianPruner,
                "SuccessiveHalving": optuna.pruners.SuccessiveHalvingPruner,
                "None": optuna.pruners.NopPruner,
                "NoPruner": optuna.pruners.NopPruner,
                "Threshold": optuna.pruners.ThresholdPruner,
                "Patient": optuna.pruners.PatientPruner,
                "Percentile": optuna.pruners.PercentilePruner,
            }
            if optuna_config["pruner"]["type"] not in pruner_options:
                raise ValueError(f"Pruner type '{optuna_config['pruner']['type']}' not recognized.")

            pruner_class = optuna_config["pruner"].pop("type")
            pruner = pruner_options[pruner_class](**optuna_config["pruner"])
            study_config = optuna_config["study"]
            url = study_config.pop("storage", None)
            storage = self._setup_optuna_storage(url, study_config) if url else None

            study = optuna.create_study(pruner=pruner, storage=storage, **study_config)
            study.optimize(objective_fn, n_trials=optuna_config["n_trials"])

            best_params = study.best_trial.params
            best_params.update(fixed_params_from_obj)

            logger.info(
                f"\nBest trial is {study.best_trial.number}."
                f"Objective value: {study.best_trial.value:.4f}"
            )
            logger.info(f"Best hyperparameters: {best_params}")

            if "build_final_model" in optuna_config and not optuna_config["build_final_model"]:
                return None, []

            logger.info("\nTraining final model with best hyperparameters...")
            trained_estimator, final_summary = self._build_and_train_model(
                params=best_params,
                x_train=x_train,
                theta_train=theta_train,
                study_direction=study_direction,
            )
            final_summary["best_params"] = best_params

        final_loss = final_summary["best_validation_loss"][0]
        if "training_log_probs" in final_summary:
            final_summary["training_loss"] = [-1.0 * x for x in final_summary["training_log_probs"]]
            final_summary["validation_loss"] = [
                -1.0 * x for x in final_summary["validation_log_probs"]
            ]
            final_summary["best_validation_loss"] = [
                -1.0 * x for x in final_summary["best_validation_log_prob"]
            ]
        else:
            final_summary["training_log_probs"] = [-1.0 * x for x in final_summary["training_loss"]]
            final_summary["validation_log_probs"] = [
                -1.0 * x for x in final_summary["validation_loss"]
            ]
            final_summary["best_validation_log_prob"] = [
                -1.0 * x for x in final_summary["best_validation_loss"]
            ]

        logger.info(f"\nFinal model trained. Best validation loss: {final_loss:.4f}")
        posterior = DirectPosterior(posterior_estimator=trained_estimator, prior=self.prior)
        posterior = EnsemblePosterior(
            posteriors=[posterior],
            weights=torch.tensor([1.0], device=self.device),
            theta_transform=posterior.theta_transform,
        )
        posterior.name = self.name
        posterior.signatures = self.signatures

        if self.out_dir:
            self._save_models(posterior, [final_summary])
        return posterior, [final_summary]

    def _build_and_train_model(
        self,
        params: Dict,
        x_train: torch.Tensor,
        theta_train: torch.Tensor,
        study_direction: str,
    ) -> Tuple[nn.Module, Dict]:
        """Builds, trains, and returns a single density estimator based on the given parameters."""
        model_name = params["model_choice"]
        logger.info(f"Building and training model: '{model_name}'")
        base_model_config = next(
            (cfg for cfg in self.net_configs if cfg["model"] == model_name), None
        )
        if base_model_config is None:
            raise ValueError(f"Base config for model '{model_name}' not found.")

        model_params = base_model_config.copy()

        trial_model_params = {
            p.split("_", 1)[1]: v for p, v in params.items() if p.startswith(model_name)
        }
        model_params.update(trial_model_params)

        valid_model_keys = base_model_config.keys()
        non_prefixed_params = {k: v for k, v in params.items() if k in valid_model_keys}
        model_params.update(non_prefixed_params)

        model_params.pop("repeats", None)

        estimator_builder = load_nde_sbi(
            self.engine, embedding_net=self.embedding_net, **model_params
        )
        if isinstance(estimator_builder, list):
            estimator_builder = estimator_builder[0]

        estimator = estimator_builder(batch_x=x_train, batch_theta=theta_train)
        estimator.to(self.device)

        optimizer_name = params.get("optimizer_choice", "Adam")
        optimizer = (Adam if optimizer_name == "Adam" else AdamW)(
            estimator.parameters(), lr=params["learning_rate"]
        )

        batch_size = params.get("training_batch_size", 32)
        dataset = data.TensorDataset(theta_train, x_train)

        validation_fraction = self.train_args.get("validation_fraction", 0.1)
        num_val = int(validation_fraction * len(dataset))
        num_train = len(dataset) - num_val

        train_indices, val_indices = torch.utils.data.random_split(
            range(len(dataset)), [num_train, num_val]
        )

        train_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(train_indices.indices)),
            sampler=SubsetRandomSampler(train_indices.indices),
            drop_last=len(train_indices.indices) > batch_size,
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(val_indices.indices)),
            sampler=SubsetRandomSampler(val_indices.indices),
            drop_last=len(val_indices.indices) > batch_size,
        )

        stop_after_epochs = params.get("stop_after_epochs", 20)
        clip_max_norm = params.get("clip_max_norm", 5.0)

        trained_estimator, _, summary = self._train_model(
            estimator,
            optimizer,
            train_loader,
            val_loader,
            study_direction=study_direction,
            stop_after_epochs=stop_after_epochs,
            clip_max_norm=clip_max_norm,
        )

        return trained_estimator, summary

    def _setup_optuna_storage(self, url: str, study_config: Dict):
        """Sets up the Optuna storage backend with retries for database connections."""
        if "://" in url:
            from sqlalchemy.exc import ProgrammingError

            retries = 5
            for attempt in range(retries):
                try:
                    storage = optuna.storages.RDBStorage(
                        url=url,
                        heartbeat_interval=study_config.pop("heartbeat_interval", 60),
                        grace_period=study_config.pop("grace_period", 120),
                        engine_kwargs={
                            "pool_pre_ping": True,
                            "pool_recycle": 3600,
                            "connect_args": {"sslmode": "disable"}
                            if "cockroachdb://" in url
                            else {},
                        },
                    )
                    logger.info(
                        f"Successfully connected to Optuna storage on attempt {attempt + 1}"
                    )
                    return storage
                except ProgrammingError as e:
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.warning(
                            f"Schema creation race condition detected (attempt {attempt + 1})"
                        )
                        if attempt < retries - 1:
                            wait_time = (2**attempt) + np.random.uniform(0, 1)
                            logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                    else:
                        raise e
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    wait_time = (2**attempt) + np.random.uniform(0, 1)
                    time.sleep(wait_time)
            raise ConnectionError("Failed to connect to Optuna storage after multiple retries.")
        else:
            from optuna.storages import JournalFileBackend, JournalStorage

            return JournalStorage(JournalFileBackend(url))

    def _calculate_objective_metric(
        self,
        trained_estimator: nn.Module,
        x_val: torch.Tensor,
        theta_val: torch.Tensor,
        metric_name: str,
        trial: optuna.Trial,
        batch_size: int = 128,
    ) -> float:
        logger.info(f"Trial {trial.number}: Calculating objective metric '{metric_name}'.")

        posterior = DirectPosterior(posterior_estimator=trained_estimator, prior=self.prior)
        logger.info(type(posterior), posterior.__dict__)

        val_dataset = data.TensorDataset(x_val, theta_val)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if metric_name in ["log_prob", "loss", "log_prob-pit"]:
            total_log_prob = 0.0
            # with torch.no_grad():
            """for x_batch, theta_batch in tqdm(val_loader, desc=f"Trial {trial.number}:
            Calculating {metric_name} for {len(x_val)} samples in {len(val_loader)} batches"):
                if theta_batch.dim() == 2:
                    theta_batch = theta_batch.unsqueeze(0)
                logger.info(posterior.log_prob_batched)
                #Wrong shape?
                logger.info(f'log_prob {theta_batch.shape}, {x_batch.shape}')
                log_prob = posterior.log_prob_batched(theta_batch, x=x_batch,
                                leakage_correction_params={'show_progress_bars':
                                logging.getLogger().level <= logging.INFO})
                total_log_prob += log_prob.sum().item()
            """

            # alternate with pure log_prob, not batched
            val_loader = data.DataLoader(val_dataset, shuffle=False)
            for x_i, theta_i in tqdm(
                val_loader,
                desc=f"Trial {trial.number}: Calculating {metric_name} for {len(x_val)} samples",
            ):
                # add a batch dimension
                if theta_i.dim() == 1:
                    theta_i = theta_i.unsqueeze(0)
                if x_i.dim() == 1:
                    x_i = x_i.unsqueeze(0)
                # logger.info(f'log_prob {theta_i.shape}, {x_i.shape}')
                log_prob = posterior.log_prob(
                    theta_i,
                    x=x_i,
                    leakage_correction_params={
                        "show_progress_bars": logging.getLogger().level <= logging.INFO,
                        "num_rejection_samples": 1000,
                    },
                )
                total_log_prob += log_prob.sum().item()

            mean_log_prob = total_log_prob / len(x_val)

            if metric_name == "log_prob":
                return mean_log_prob
            if metric_name == "loss":
                return -mean_log_prob
            if metric_name == "log_prob-pit":
                score = mean_log_prob
                all_samples = []
                with torch.no_grad():
                    for x_batch, _ in tqdm(
                        val_loader, desc=f"Trial {trial.number}: Calculating {metric_name}"
                    ):
                        samples = posterior.sample_batched(
                            (1000,), x=x_batch, show_progress_bars=False
                        )
                        all_samples.append(samples)
                all_samples = torch.cat(all_samples, dim=1).cpu().numpy()

                pit = np.empty(len(x_val))
                for i in range(len(x_val)):
                    pit[i] = np.mean(all_samples[:, i, :] < theta_val[i].cpu().numpy())

                pit = np.sort(pit)
                if pit[-1] > 0:
                    pit /= pit[-1]
                dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
                score += -0.5 * np.log(dpit_max + 1e-9)  # Add epsilon for stability
                return score

        elif metric_name == "tarp":
            num_samples = 1000
            all_samples = []
            with torch.no_grad():
                for x_batch, _ in tqdm(
                    val_loader, desc=f"Trial {trial.number}: Calculating {metric_name}"
                ):
                    samples = posterior.sample_batched(
                        (num_samples,), x=x_batch, show_progress_bars=False
                    )
                    all_samples.append(samples)
            all_samples = torch.cat(all_samples, dim=1)

            ecp, _ = tarp.get_tarp_coverage(
                all_samples, theta_val, norm=True, bootstrap=True, num_bootstrap=200
            )
            tarp_val = torch.mean(torch.from_numpy(ecp[:, ecp.shape[1] // 2])).to(self.device)
            return float(abs(tarp_val - 0.5))
        elif callable(metric_name):
            return metric_name(posterior, x_val, theta_val)
        else:
            raise ValueError(
                f"Trial {trial.number}: '{metric_name}' not recognized for custom objective."
            )

    @staticmethod
    def _train_model(
        density_estimator: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        max_num_epochs: int = 2**31 + 1,
        stop_after_epochs: int = 20,
        trial: Optional[optuna.Trial] = None,
        study_direction: str = "minimize",
        clip_max_norm: Optional[float] = 5.0,
    ) -> Tuple[nn.Module, float, Dict]:
        """Improved training loop following SBI conventions.

        Key improvements:
        1. Proper loss accumulation (sum then divide by total samples)
        2. Consistent tensor handling with .sum() for batch losses
        3. Better convergence tracking
        4. Gradient clipping support
        5. Proper model state management
        """
        best_val_loss = float("inf")
        epochs_since_improvement = 0
        best_model_state_dict = None
        train_log, val_log = [], []
        epoch = 0

        # Helper function to check convergence
        def _converged(current_epoch: int, stop_after_epochs: int) -> bool:
            return epochs_since_improvement >= stop_after_epochs

        start_time = time.time()
        while epoch <= max_num_epochs and not _converged(epoch, stop_after_epochs):
            # Training phase
            density_estimator.train()
            train_loss_sum = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                # Unpack batch (handling potential mask dimension)
                if len(batch) == 2:
                    theta_batch, x_batch = batch
                    masks_batch = None
                else:
                    theta_batch, x_batch, masks_batch = batch

                # Calculate loss - following SBI convention where loss() returns per-sample losses
                if hasattr(density_estimator, "loss"):
                    # Standard SBI estimator
                    if masks_batch is not None:
                        train_losses = density_estimator.loss(theta_batch, x_batch, masks_batch)
                    else:
                        train_losses = density_estimator.loss(theta_batch, x_batch)
                else:
                    # Fallback for custom estimators
                    train_losses = -density_estimator.log_prob(theta_batch, context=x_batch)

                # Mean for backprop, but sum for accumulation (following SBI pattern)
                train_loss = torch.mean(train_losses)
                train_loss_sum += train_losses.sum().item()  # Sum of individual losses

                train_loss.backward()

                # Gradient clipping (SBI uses 5.0 by default)
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        density_estimator.parameters(), max_norm=clip_max_norm
                    )

                optimizer.step()

            epoch += 1

            # Calculate average loss per sample (not per batch)
            train_loss_average = train_loss_sum / (len(train_loader) * train_loader.batch_size)
            train_log.append(train_loss_average)

            # Validation phase
            density_estimator.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    # Unpack batch
                    if len(batch) == 2:
                        theta_batch, x_batch = batch
                        masks_batch = None
                    else:
                        theta_batch, x_batch, masks_batch = batch

                    # Calculate validation losses
                    if hasattr(density_estimator, "loss"):
                        if masks_batch is not None:
                            val_losses = density_estimator.loss(theta_batch, x_batch, masks_batch)
                        else:
                            val_losses = density_estimator.loss(theta_batch, x_batch)
                    else:
                        val_losses = -density_estimator.log_prob(theta_batch, context=x_batch)

                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples
            current_val_loss = val_loss_sum / (len(val_loader) * val_loader.batch_size)
            val_log.append(current_val_loss)

            # Early stopping and best model tracking
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_since_improvement = 0
                best_model_state_dict = deepcopy(density_estimator.state_dict())
            else:
                epochs_since_improvement += 1

            if trial:
                if study_direction == "maximize":
                    # If maximizing, report negative loss (e.g. log_prob)
                    trial.report(-current_val_loss, epoch - 1)
                else:
                    trial.report(current_val_loss, epoch - 1)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number}: Pruned at epoch {epoch}.")
                    raise optuna.TrialPruned()
            else:
                time_elapsed = time.time() - start_time
                update_plot(train_log, val_log, epoch=epoch, time_elapsed=time_elapsed)

                """logger.info(
                    f"Epoch {epoch}: TL: {train_loss_average:.3f}, "
                    f"VL: {current_val_loss:.3f}, "
                    f"Best VL: {best_val_loss:.3f}, "
                    f"ESI: {epochs_since_improvement}/{stop_after_epochs}"
                )"""

        # Restore best model state
        if best_model_state_dict is not None:
            density_estimator.load_state_dict(best_model_state_dict)

        # Clean up gradients (following SBI pattern)
        density_estimator.zero_grad(set_to_none=True)

        if trial is None:
            print("\nTraining complete.")
            _exit_alternate_screen()
            # final plot
            time_elapsed = time.time() - start_time
            update_plot(
                train_log, val_log, epoch=epoch, time_elapsed=time_elapsed, alt_screen=False
            )

        # Create summary following SBI conventions
        summary = {
            "training_loss": train_log,
            "validation_loss": val_log,
            "best_validation_loss": [best_val_loss],
            "epochs_trained": [epoch],
            "converged": epochs_since_improvement >= stop_after_epochs,
        }
        temp_str = "" if trial is None else f"Trial {trial.number}: "
        logger.info(
            f"{temp_str}Training completed after {epoch} epochs. "
            f"Best validation loss: {best_val_loss:.6f}"
        )

        return density_estimator, best_val_loss, summary

    def _create_objective(
        self,
        x_train: torch.Tensor,
        theta_train: torch.Tensor,
        x_val: Optional[torch.Tensor],
        theta_val: Optional[torch.Tensor],
        study_direction: str = "minimize",
    ) -> Callable[[optuna.Trial], float]:
        """Creates the Optuna objective function for hyperparameter optimization.

        Improved objective function with better data handling.
        Only creates trial dimensions for parameters that actually need optimization.
        """
        optuna_config = self.train_args["optuna"]
        search_space = optuna_config["search_space"]
        max_num_epochs = self.train_args.get("max_resource", 2**32)  # Default to a very high number
        fixed_params = {}

        def objective(trial: optuna.Trial) -> float:
            # Only suggest model choice if there are multiple options
            if len(search_space["model_choice"]) > 1:
                model_name = trial.suggest_categorical("model_choice", search_space["model_choice"])
            else:
                model_name = search_space["model_choice"][0]
                fixed_params["model_choice"] = model_name

            base_model_config = next(
                (cfg for cfg in self.net_configs if cfg["model"] == model_name), None
            )
            if base_model_config is None:
                raise ValueError(f"Base config for model '{model_name}' not found.")

            trial_model_params = base_model_config.copy()
            for param, settings in search_space["models"][model_name].items():
                param_name = f"{model_name}_{param}"
                try:
                    if settings["type"] == "categorical":
                        # Only create trial dimension if there are multiple choices
                        if len(settings["choices"]) > 1:
                            trial_model_params[param] = trial.suggest_categorical(
                                param_name, settings["choices"]
                            )
                        else:
                            trial_model_params[param] = settings["choices"][0]
                    elif settings["type"] == "int":
                        # Only create trial dimension if low != high
                        if "value" not in settings.keys():
                            trial_model_params[param] = trial.suggest_int(
                                param_name, int(settings["low"]), int(settings["high"])
                            )
                        else:
                            trial_model_params[param] = settings["value"]
                            fixed_params[param] = settings["value"]
                except Exception as e:
                    raise Exception(
                        f"Model configuration not understood for {param}, {settings} {e}"
                    )

            trial_model_params.pop("repeats", None)

            # Get other hyperparameters - only suggest if there's a range to optimize
            lr_config = search_space["learning_rate"]
            if "value" not in lr_config.keys():
                lr = trial.suggest_float(
                    "learning_rate",
                    float(lr_config["low"]),
                    float(lr_config["high"]),
                    log=lr_config.get("log", False),
                )
            else:
                lr = lr_config["value"]
                fixed_params["learning_rate"] = lr

            # Optimizer choice
            optimizer_choices = search_space.get("optimizer_choice", ["Adam"])
            if len(optimizer_choices) > 1:
                optimizer_name = trial.suggest_categorical("optimizer_choice", optimizer_choices)
            else:
                optimizer_name = optimizer_choices[0]

            # Batch size
            batch_size_config = search_space.get("training_batch_size", {})
            if batch_size_config.get("type") == "categorical":
                choices = batch_size_config.get("choices", [32])
                if len(choices) > 1:
                    batch_size = trial.suggest_categorical("training_batch_size", choices)
                else:
                    batch_size = choices[0]
                    fixed_params["training_batch_size"] = batch_size
            elif batch_size_config.get("type") == "int":
                if "value" not in batch_size_config.keys():
                    batch_size = trial.suggest_int(
                        "training_batch_size", batch_size_config["low"], batch_size_config["high"]
                    )
                else:
                    batch_size = batch_size_config["value"]
                    fixed_params["training_batch_size"] = batch_size
            else:
                batch_size = 32  # Default
                fixed_params["training_batch_size"] = batch_size

            # Stop after epochs
            stop_config = search_space.get("stop_after_epochs", {})
            if stop_config and "value" not in stop_config.keys():
                stop_after_epochs = trial.suggest_int(
                    "stop_after_epochs", stop_config["low"], stop_config["high"]
                )
            else:
                stop_after_epochs = stop_config.get("low", 20)  # Default to 20
                fixed_params["stop_after_epochs"] = stop_after_epochs

            # Gradient clipping
            clip_config = search_space.get("clip_max_norm", {})
            if clip_config and "value" not in clip_config.keys():
                clip_max_norm = trial.suggest_float(
                    "clip_max_norm", clip_config["low"], clip_config["high"]
                )
            else:
                clip_max_norm = clip_config.get("value", 5.0)  # Default to 5.0
                fixed_params["clip_max_norm"] = clip_max_norm

            # Build density estimator
            density_estimator_builder = load_nde_sbi(
                self.engine, embedding_net=self.embedding_net, **trial_model_params
            )

            for key, param in fixed_params.items():
                trial.set_user_attr(key, param)

            if isinstance(density_estimator_builder, list):
                density_estimator_builder = density_estimator_builder[0]
            logger.debug(
                f"Density Estimator: {density_estimator_builder}, "
                f"type: {type(density_estimator_builder)}, "
                f"params: {density_estimator_builder.__dict__}"
            )
            density_estimator = density_estimator_builder(batch_x=x_train, batch_theta=theta_train)
            density_estimator.to(self.device)
            logger.debug(
                f"{density_estimator}, type: {type(density_estimator)}, params: {density_estimator.__dict__}"  # noqa: E501
            )

            optimizer = (Adam if optimizer_name == "Adam" else AdamW)(
                density_estimator.parameters(), lr=lr
            )

            param_str = (
                f"{trial_model_params=:}".replace("'", "")
                .replace("{", "")
                .replace("}", "")
                .split("=")[-1]
                .replace(":", " =")
            )
            logger.info(
                f"Trial {trial.number}: model='{model_name}', lr={lr:.2e}, "
                f"optimizer={optimizer_name}, batch_size={batch_size}, "
                f"clip_norm={clip_max_norm}, {param_str}"
            )

            # Create data loaders with proper dataset structure
            dataset = data.TensorDataset(theta_train, x_train)

            # Use SBI-style validation fraction (0.1 means 10% validation, 90% training)
            validation_fraction = self.train_args.get("validation_fraction", 0.1)
            num_val = int(validation_fraction * len(dataset))
            num_train = len(dataset) - num_val

            train_indices, val_indices = torch.utils.data.random_split(
                range(len(dataset)), [num_train, num_val]
            )

            train_dl = data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_indices.indices),
                drop_last=len(train_indices) > batch_size,  # Avoid empty batches
            )
            val_dl = data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(val_indices.indices),
                drop_last=len(val_indices) > batch_size,
            )

            trial.set_user_attr("train_val_size", len(val_indices))
            trial.set_user_attr("train_size", len(train_indices))
            start_train = datetime.now()
            trial.set_user_attr("start_time", str(start_train))

            # Train with improved loop
            trained_estimator, best_val_loss, _ = self._train_model(
                density_estimator,
                optimizer,
                train_dl,
                val_dl,
                max_num_epochs=max_num_epochs,
                trial=trial,
                study_direction=study_direction,
                stop_after_epochs=stop_after_epochs,
                clip_max_norm=clip_max_norm,
            )
            end_train = datetime.now()
            trial.set_user_attr("train_time", str(start_train - end_train))

            # Use external validation data if available
            if x_val is not None and theta_val is not None:
                metric_name = optuna_config["objective"]["metric"]
                best_val_loss = self._calculate_objective_metric(
                    trained_estimator,
                    x_val,
                    theta_val,
                    metric_name,
                    trial=trial,
                )
                trial.set_user_attr("metric", metric_name)
                trial.set_user_attr("val_size", x_val.shape)

            trial.set_user_attr("validate_time", str(datetime.now() - end_train))

            return best_val_loss

        return objective, fixed_params


class Interval(constraints.Constraint):
    """Constrain to a real interval `[lower_bound, upper_bound]`."""

    def __init__(self, lower_bound, upper_bound, validate_func=None):
        """Initialize the Interval constraint."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.validate_func = validate_func
        self.depth = 0
        super().__init__()

    def check(self, value):
        """Check if the value is within the interval."""
        if self.validate_func is not None and self.depth == 0:
            self.validate_func(value)
        x = (self.lower_bound <= value) & (value <= self.upper_bound)
        return x

    def __repr__(self):
        """String representation of the constraint."""
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        return fmt_string


biject_to.register(Interval, _transform_to_interval)
transform_to.register(Interval, _transform_to_interval)
# IndependentInterval = constraints._IndependentConstraint(Interval([1], [10], ['a']), 1)

# biject_to.register(IndependentInterval, _biject_to_independent)
# transform_to.register(IndependentInterval, _transform_to_independent)


class CustomUniform(Distribution):
    r"""Custom Uniform distribution with enhanced validation.

    A custom Uniform distribution that accepts a list of parameter names for
    enhanced validation error messages, especially for batched data.

    It generates uniformly distributed random samples from the half-open
    interval ``[low, high)``.

    Args:
        low (float or Tensor): Lower range (inclusive).
        high (float or Tensor): Upper range (exclusive).
        name_list (List[str]): A list of names for each parameter dimension.
                               Its length must match the number of parameters.
        validate_args (bool, optional): Whether to validate arguments.
                                        Defaults to ``Distribution._validate_args``.
        report_threshold (float, optional): Threshold for reporting validation errors.
                                              Defaults to 0.1.
    """

    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    has_rsample = True

    @property
    def mean(self) -> torch.Tensor:
        """Returns the mean of the distribution."""
        return (self.high + self.low) / 2

    @property
    def mode(self) -> torch.Tensor:
        """Returns the mode of the distribution. For a uniform distribution, this is undefined."""
        return torch.full_like(self.high, float("nan"))

    @property
    def stddev(self) -> torch.Tensor:
        """Returns the standard deviation of the distribution."""
        return (self.high - self.low) / (12**0.5)

    @property
    def variance(self) -> torch.Tensor:
        """Returns the variance of the distribution."""
        return (self.high - self.low).pow(2) / 12

    def __init__(
        self,
        low: Union[float, torch.Tensor],
        high: Union[float, torch.Tensor],
        name_list: List[str],
        verbose: bool = True,
        validate_args: Optional[bool] = None,
        report_threshold: float = 0.1,
    ):
        """Initializes the CustomUniform distribution."""
        self.low, self.high = broadcast_all(low, high)
        self.name_list = name_list
        self.verbose = verbose
        self.report_threshold = report_threshold

        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()

        # Ensure name_list matches parameter dimensions
        num_params = self.low.shape[-1] if self.low.dim() > 0 else 1
        if len(self.name_list) != num_params:
            raise ValueError(
                f"Length of name_list ({len(self.name_list)}) must match the "
                f"number of parameters ({num_params})."
            )

        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape: _size, _instance=None) -> "CustomUniform":
        """Expands the distribution to the desired batch shape."""
        new = self._get_checked_instance(CustomUniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        new.name_list = self.name_list
        super(CustomUniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """Returns the support constraint with validation if verbose is True."""
        return Interval(self.low, self.high, self._validate_sample if self.verbose else None)

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Generates a sample_shape shaped reparameterized sample."""
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability density function."""
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function."""
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        """Inverse of the CDF function."""
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self) -> torch.Tensor:
        """Returns the differential entropy of the distribution."""
        return torch.log(self.high - self.low)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def _original_support(self) -> constraints.Constraint:
        """Returns the original support constraint without validation."""
        return constraints.interval(self.low, self.high)

    def _validate_sample(self, value: torch.Tensor) -> None:
        """Custom argument validation with aggregated errors for out-of-support parameters."""
        if not isinstance(value, torch.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        # use a copy of the value to avoid modifying the original tensor
        value = value.clone()
        # Standard shape validation from the base class
        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError(
                f"The right-most size of value must match event_shape: {value.size()} vs {self._event_shape}."  # noqa: E501
            )
        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    f"Value is not broadcastable with batch_shape+event_shape: {actual_shape} vs {expected_shape}."  # noqa: E501
                )
        # Custom support validation with aggregated summary
        is_in_support = self._original_support.check(value)
        support_fraction = np.sum(is_in_support.cpu().numpy()) / is_in_support.numel()
        if support_fraction < self.report_threshold:
            error_messages = []
            param_dim = -1  # Assumes parameters are in the last dimension
            num_params = is_in_support.shape[param_dim]

            for i in range(num_params):
                param_is_in_support = is_in_support[..., i]
                if not param_is_in_support.all():
                    # Count how many samples in the batch are invalid for this parameter
                    num_invalid = (~param_is_in_support).sum().item()
                    total_samples = param_is_in_support.numel()
                    param_name = self.name_list[i]
                    low_bound = self.low[..., i].item()
                    high_bound = self.high[..., i].item()
                    error_messages.append(
                        f"  - Parameter '{param_name}' "
                        f"(support [{low_bound:.2f}, {high_bound:.2f})): "
                        f"{num_invalid}/{total_samples} samples are out of support."
                    )
            # out of total samples(how many have support in every parameter)
            total = torch.all(is_in_support, axis=-1).sum().float().mean() / total_samples
            error_messages.append(
                f"  - In total {total * 100:.2f}% samples are within support across all parameters."
            )
            # Log a single aggregated message
            if error_messages:
                full_error_message = (
                    "Checked sample acceptance within batched sample. Summary:\n"
                    + "\n".join(error_messages)
                )
                logging.warning(full_error_message)


class CustomIndependentUniform(CustomUniform):
    r"""A wrapper around CustomUniform to create an independent distribution."""

    def __init__(self, *args, device="cpu", **kwargs):
        """A wrapper around CustomUniform to create an independent distribution."""
        args = [
            torch.as_tensor(v, dtype=torch.float32, device=device)
            if isinstance(v, (float, int, torch.Tensor, np.ndarray))
            else v
            for v in args
        ]
        kwargs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            if isinstance(v, (float, int, torch.Tensor, np.ndarray))
            else v
            for k, v in kwargs.items()
        }

        super().__init__(*args, **kwargs)
