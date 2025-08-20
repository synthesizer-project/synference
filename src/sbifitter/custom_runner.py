"""Custom runner like LtU-ILI's SBIRunner, but with Optuna-based hyperparam optimization."""

from copy import deepcopy
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import sys

import optuna
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import yaml
import logging
from ili.dataloaders.loaders import StaticNumpyLoader
from ili.inference import SBIRunner
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from torch.distributions import Distribution
from torch.optim import Adam, AdamW
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

try:  # sbi > 0.22.0
    from sbi.inference.posteriors import EnsemblePosterior
except ImportError:  # sbi < 0.22.0
    from sbi.utils.posterior_ensemble import (
        NeuralPosteriorEnsemble as EnsemblePosterior,
    )

from ili.dataloaders import _BaseLoader
from ili.utils import load_from_config, load_nde_sbi

from . import logger

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
    ) -> "EnsemblePosterior":
        """Run the custom SBI inference with Optuna hyperparameter optimization.

        Args:
            loader (_BaseLoader): The data loader containing training data.
            validation_loader (_BaseLoader, optional): The data loader containing validation data.
                If not provided, it will use the training loader for validation.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
            optuna.samplers.TPESampler(seed=seed)

        optuna_config = self.train_args["optuna"]

        x_train = torch.from_numpy(loader.get_all_data()).float().to(self.device)
        theta_train = torch.from_numpy(loader.get_all_parameters()).float().to(self.device)

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
        objective_fn, fixed_params = self._create_objective(x_train, theta_train, x_val, theta_val,
                                              study_direction)

        logger.info("Starting Optuna hyperparameter search...")
        if "pruner" in optuna_config:
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
            raise ValueError(
                f"Pruner type '{optuna_config['pruner']['type']}' not recognized. "
                f"Available options: {list(pruner_options.keys())}"
            )

        pruner_class = optuna_config["pruner"].pop("type")
        pruner = pruner_options[pruner_class](**optuna_config["pruner"])

        study_config = optuna_config["study"]
        url = study_config.pop("storage", None)

        retries = 5
        import time
        from sqlalchemy.exc import ProgrammingError
        if '://' in url:
            for attempt in range(retries):
                try:
                    storage = (
                        optuna.storages.RDBStorage(url=url,
                                                heartbeat_interval=study_config.pop("heartbeat_interval", 60),
                                                grace_period=study_config.pop("grace_period", 120),
                                                engine_kwargs={
                                                'pool_pre_ping': True,
                                                'pool_recycle': 3600,
                                                'connect_args': {'sslmode': 'disable'} if 'cockroachdb://' in url else {}
                                            })
                        if url else None
                    )
                    logger.info(f"Successfully connected to Optuna storage on attempt {attempt + 1}")
                    break

                
                except ProgrammingError as e:
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.warning(f"Schema creation race condition detected (attempt {attempt + 1})")

                        if attempt < retries - 1:
                            # Wait with exponential backoff + jitter
                            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                            logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        
                    else:
                        raise e
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    time.sleep(wait_time)
        else:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            storage = JournalStorage(JournalFileBackend(url))

            
        study = optuna.create_study(pruner=pruner, storage=storage, **study_config)
        study.optimize(objective_fn, n_trials=optuna_config["n_trials"])

        best_params = study.best_trial.params
        
        logger.info(f"\nBest trial is {study.best_trial.number}. Objective value: {study.best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")

        if "build_final_model" in optuna_config and not optuna_config["build_final_model"]:
            return None, []

        best_params.update(fixed_params)
        logger.info("\nTraining final model with best hyperparameters...")
        final_model_name = best_params["model_choice"]

        base_model_config = next(
            (cfg for cfg in self.net_configs if cfg["model"] == final_model_name), None
        )
        if base_model_config is None:
            raise ValueError(f"Base config for model '{final_model_name}' not found.")

        final_model_params = base_model_config.copy()
        trial_model_params = {
            p.split("_", 1)[1]: v for p, v in best_params.items() if p.startswith(final_model_name)
        }
        final_model_params.update(trial_model_params)

        final_model_params.pop("repeats", None)  # Ensure we build a single model for the trial

        final_estimator = load_nde_sbi(
            self.engine, embedding_net=self.embedding_net, **final_model_params
        )
        if isinstance(final_estimator, list):
            final_estimator = final_estimator[0]

        final_estimator = final_estimator(batch_x=x_train, batch_theta=theta_train)
        final_estimator.to(self.device)

        optimizer = (Adam if best_params["optimizer_choice"] == "Adam" else AdamW)(
            final_estimator.parameters(), lr=best_params["learning_rate"]
        )

        dataset = data.TensorDataset(theta_train, x_train)
        num_train = int((1 - self.train_args.get("validation_fraction", 0.9)) * len(dataset))
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(dataset)), [num_train, len(dataset) - num_train]
        )
        train_loader = data.DataLoader(
            dataset,
            batch_size=min(best_params["training_batch_size"], len(train_indices)),
            sampler=SubsetRandomSampler(train_indices.indices),
            drop_last=True,
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(best_params["training_batch_size"], len(val_indices)),
            sampler=SubsetRandomSampler(val_indices.indices),
            drop_last=True,
        )

        stop_after_epochs = best_params.get("stop_after_epochs", 20)

        trained_estimator, final_loss, final_summary = self._train_model(
            final_estimator,
            optimizer,
            train_loader,
            val_loader,
            study_direction=study_direction,
            stop_after_epochs=stop_after_epochs,
        )

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

        final_summary["best_params"] = best_params
        logger.info(f"\nFinal model trained. Best training validation loss: {final_loss:.4f}")

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


    def _calculate_objective_metric(
        self,
        trained_estimator: nn.Module,
        x_val: torch.Tensor,
        theta_val: torch.Tensor,
        metric_name: str,
        trial: optuna.Trial,
    ) -> float:
        logger.info(f"Trial {trial.number}: Calculating objective metric '{metric_name}'.")

        posterior = DirectPosterior(posterior_estimator=trained_estimator, prior=self.prior)

        if metric_name == "log_prob":
            # add a leading dimension to theta_val if needed
            if theta_val.shape[0] != 1:
                theta_val = theta_val.unsqueeze(0)

            logger.info(f'Theta: {theta_val.shape}, X: {x_val.shape}')

            log_prob = posterior.log_prob_batched(theta_val, x=x_val, leakage_correction_params={'show_progress_bars':logger.level <= logging.INFO})
            return float(log_prob.mean())
        elif metric_name == "loss":
            # add a leading dimension to theta_val if needed
            if theta_val.shape[0] != 1:
                theta_val = theta_val.unsqueeze(0)

            log_prob = posterior.log_prob_batched(theta_val, x=x_val, leakage_correction_params={'show_progress_bars':logger.level <= logging.INFO})
            return -1*float(log_prob.mean())
        if metric_name == "log_prob-pit":
            # add a leading dimension to theta_val if needed
            if theta_val.shape[0] != 1:
                theta_val = theta_val.unsqueeze(0)

            score = float(posterior.log_prob_batched(theta_val, x=x_val, leakage_correction_params={'show_progress_bars':logger.level <= logging.INFO}).mean())
            samples = posterior.sample_batched((1000,), x=x_val, show_progress_bars=False).cpu().numpy()
            pit = np.empty(len(x_val))
            for i in range(len(x_val)):
                pit[i] = np.mean(samples[i] < x_val[i])

            pit = np.sort(pit)
            pit /= pit[-1]
            dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
            score += -0.5 * np.log(dpit_max)
            return score

        elif metric_name == "tarp":
            import tarp

            num_samples = 1000
            samples = posterior.sample_batched((num_samples,), x=x_val, show_progress_bars=False)
            ecp, _ = tarp.get_tarp_coverage(
                samples, theta_val, norm=True, bootstrap=True, num_bootstrap=200
            )

            tarp_val = torch.mean(torch.from_numpy(ecp[:, ecp.shape[1] // 2])).to(self.device)
            return float(abs(tarp_val - 0.5))
        elif callable(metric_name):
            return metric_name(posterior, x_val, theta_val)
        else:
            raise ValueError(f"Trial {trial.number}: '{metric_name}' not recognized for custom objective.")

    @staticmethod
    def _train_model(
        density_estimator: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        max_num_epochs: int = 2**31+1,
        stop_after_epochs: int = 20,
        trial: Optional[optuna.Trial] = None,
        study_direction: str = "minimize",
        clip_max_norm: Optional[float] = 5.0,
    ) -> Tuple[nn.Module, float, Dict]:
        """
        Improved training loop following SBI conventions.
        
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
                if hasattr(density_estimator, 'loss'):
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
                    if hasattr(density_estimator, 'loss'):
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

        # Restore best model state
        if best_model_state_dict is not None:
            density_estimator.load_state_dict(best_model_state_dict)
        
        # Clean up gradients (following SBI pattern)
        density_estimator.zero_grad(set_to_none=True)

        # Create summary following SBI conventions
        summary = {
            "training_loss": train_log,
            "validation_loss": val_log,
            "best_validation_loss": [best_val_loss],
            "epochs_trained": [epoch],
            "converged": epochs_since_improvement >= stop_after_epochs
        }
        
        logger.info(f"Trial {trial.number}: Training completed after {epoch} epochs. "
                    f"Best validation loss: {best_val_loss:.6f}")
        
        return density_estimator, best_val_loss, summary


    def _create_objective(
        self,
        x_train: torch.Tensor,
        theta_train: torch.Tensor,
        x_val: Optional[torch.Tensor],
        theta_val: Optional[torch.Tensor],
        study_direction: str = "minimize",
    ) -> Callable[[optuna.Trial], float]:
        """
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
                        if 'value' not in settings.keys():
                            trial_model_params[param] = trial.suggest_int(
                                param_name, int(settings["low"]), int(settings["high"])
                            )
                        else:
                            trial_model_params[param] = settings["value"]
                            fixed_params[param] = settings["value"]
                except Exception:
                    raise Exception(f'Model configuration not understood for {param}, {settings} {e}')

            trial_model_params.pop("repeats", None)

            # Get other hyperparameters - only suggest if there's a range to optimize
            lr_config = search_space["learning_rate"]
            if 'value' not in lr_config.keys():
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
                if 'value' not in batch_size_config.keys():
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
            if stop_config and 'value' not in stop_config.keys():
                stop_after_epochs = trial.suggest_int(
                    "stop_after_epochs", stop_config["low"], stop_config["high"]
                )
            else:
                stop_after_epochs = stop_config.get("low", 20)  # Default to 20
                fixed_params["stop_after_epochs"] = stop_after_epochs
            
            # Gradient clipping
            clip_config = search_space.get("clip_max_norm", {})
            if clip_config and 'value' not in clip_config.keys():
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

            density_estimator = density_estimator_builder(batch_x=x_train, batch_theta=theta_train)
            density_estimator.to(self.device)
            
            optimizer = (Adam if optimizer_name == "Adam" else AdamW)(
                density_estimator.parameters(), lr=lr
            )

            param_str = f'{trial_model_params=:}'.replace("'", "").replace("{","").replace("}","").split('=')[-1].replace(':', ' =')
            logger.info(f"Trial {trial.number}: model='{model_name}', lr={lr:.2e}, "
                    f"optimizer={optimizer_name}, batch_size={batch_size}, "
                    f"clip_norm={clip_max_norm}, {param_str}")

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
                drop_last=len(train_indices) > batch_size  # Avoid empty batches
            )
            val_dl = data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=SubsetRandomSampler(val_indices.indices),
                drop_last=len(val_indices) > batch_size
            )

            trial.set_user_attr('train_val_size', len(val_indices))
            trial.set_user_attr('train_size', len(train_indices))
            start_train = datetime.now()
            trial.set_user_attr('start_time',  str(start_train))

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
                clip_max_norm=clip_max_norm
            )
            end_train = datetime.now()
            trial.set_user_attr('train_time', str(start_train - end_train))

            # Use external validation data if available
            if x_val is not None and theta_val is not None:
                metric_name = optuna_config["objective"]["metric"]
                best_val_loss = self._calculate_objective_metric(
                    trained_estimator, x_val, theta_val, metric_name, trial=trial,
                )
                trial.set_user_attr('metric', metric_name)
                trial.set_user_attr('val_size',x_val.shape)

            trial.set_user_attr('validate_time', str(datetime.now() - end_train))
            
            return best_val_loss

        return objective, fixed_params