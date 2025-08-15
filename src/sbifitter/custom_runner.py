"""Custom runner like LtU-ILI's SBIRunner, but with Optuna-based hyperparam optimization."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import optuna
import torch
import torch.nn as nn
import yaml
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

logging.basicConfig(level=logging.INFO)


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

        objective_fn = self._create_objective(x_train, theta_train, x_val, theta_val)

        logging.info("Starting Optuna hyperparameter search...")
        if "pruner" in optuna_config:
            pruner_options = {
                "Hyperband": optuna.pruners.HyperbandPruner,
                "Median": optuna.pruners.MedianPruner,
                "SuccessiveHalving": optuna.pruners.SuccessiveHalvingPruner,
            }
        if optuna_config["pruner"]["type"] not in pruner_options:
            raise ValueError(
                f"Pruner type '{optuna_config['pruner']['type']}' not recognized. "
                f"Available options: {list(pruner_options.keys())}"
            )
            pruner_class = optuna_config["pruner"].pop("type")
            pruner = pruner_options[pruner_class](**optuna_config["pruner"])
        else:
            pruner = None

        study_config = optuna_config["study"]
        storage = (
            optuna.storages.RDBStorage(url=study_config.pop("storage"))
            if "storage" in study_config
            else None
        )
        study = optuna.create_study(pruner=pruner, storage=storage, **study_config)
        study.optimize(objective_fn, n_trials=optuna_config["n_trials"])

        best_params = study.best_trial.params
        logging.info(f"\nBest trial complete. Objective value: {study.best_trial.value:.4f}")
        logging.info(f"Best hyperparameters: {best_params}")

        logging.info("\nTraining final model with best hyperparameters...")
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
            batch_size=best_params["training_batch_size"],
            sampler=SubsetRandomSampler(train_indices.indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=best_params["training_batch_size"],
            sampler=SubsetRandomSampler(val_indices.indices),
        )

        trained_estimator, final_loss, final_summary = self._train_model(
            final_estimator,
            optimizer,
            train_loader,
            val_loader,
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
        logging.info(f"\nFinal model trained. Best training validation loss: {final_loss:.4f}")

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

    def _create_objective(
        self,
        x_train: torch.Tensor,
        theta_train: torch.Tensor,
        x_val: Optional[torch.Tensor],
        theta_val: Optional[torch.Tensor],
    ) -> Callable[[optuna.Trial], float]:
        optuna_config = self.train_args["optuna"]
        search_space = optuna_config["search_space"]
        max_num_epochs = optuna_config["pruner"]["max_resource"]

        def objective(trial: optuna.Trial) -> float:
            model_name = trial.suggest_categorical("model_choice", search_space["model_choice"])

            base_model_config = next(
                (cfg for cfg in self.net_configs if cfg["model"] == model_name), None
            )
            if base_model_config is None:
                raise ValueError(f"Base config for model '{model_name}' not found.")

            trial_model_params = base_model_config.copy()
            for param, settings in search_space["models"][model_name].items():
                param_name = f"{model_name}_{param}"
                if settings["type"] == "categorical":
                    trial_model_params[param] = trial.suggest_categorical(
                        param_name, settings["choices"]
                    )
                elif settings["type"] == "int":
                    trial_model_params[param] = trial.suggest_int(
                        param_name, int(settings["low"]), int(settings["high"])
                    )

            trial_model_params.pop("repeats", None)  # Ensure we build a single model for the trial

            density_estimator_builder = load_nde_sbi(
                self.engine, embedding_net=self.embedding_net, **trial_model_params
            )
            if isinstance(density_estimator_builder, list):
                density_estimator_builder = density_estimator_builder[
                    0
                ]  # Use the first model if ensemble

            lr_config = search_space["learning_rate"]
            lr = trial.suggest_float(
                "learning_rate",
                float(lr_config["low"]),
                float(lr_config["high"]),
                log=lr_config["log"],
            )
            optimizer_name = trial.suggest_categorical(
                "optimizer_choice", search_space["optimizer_choice"]
            )

            bs_config = search_space["training_batch_size"]
            if bs_config["type"] == "categorical":
                batch_size = trial.suggest_categorical("training_batch_size", bs_config["choices"])
            elif bs_config["type"] == "int":
                batch_size = trial.suggest_int(
                    "training_batch_size", bs_config["low"], bs_config["high"]
                )

            density_estimator = density_estimator_builder(batch_x=x_train, batch_theta=theta_train)
            density_estimator.to(self.device)
            optimizer = (Adam if optimizer_name == "Adam" else AdamW)(
                density_estimator.parameters(), lr=lr
            )

            dataset = data.TensorDataset(theta_train, x_train)
            num_train = int((1 - self.train_args.get("validation_fraction", 0.9)) * len(dataset))
            train_indices, val_indices = torch.utils.data.random_split(
                range(len(dataset)), [num_train, len(dataset) - num_train]
            )
            train_dl = data.DataLoader(
                dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices.indices)
            )
            val_dl = data.DataLoader(
                dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices.indices)
            )

            trained_estimator, best_train_val_loss, _ = self._train_model(
                density_estimator, optimizer, train_dl, val_dl, max_num_epochs, trial=trial
            )

            if x_val is not None and theta_val is not None:
                metric_name = optuna_config["objective"]["metric"]
                return self._calculate_objective_metric(
                    trained_estimator, x_val, theta_val, metric_name
                )
            return best_train_val_loss

        return objective

    def _calculate_objective_metric(
        self,
        trained_estimator: nn.Module,
        x_val: torch.Tensor,
        theta_val: torch.Tensor,
        metric_name: str,
    ) -> float:
        logging.info(f"Calculating objective metric '{metric_name}' for trial.")

        posterior = DirectPosterior(posterior_estimator=trained_estimator, prior=self.prior)

        if metric_name == "log_prob":
            # add a leading dimension to theta_val if needed
            if theta_val.shape[0] != 1:
                theta_val = theta_val.unsqueeze(0)

            log_prob = posterior.log_prob_batched(theta_val, x=x_val)
            return -float(log_prob.mean())
        elif metric_name == "tarp":
            import tarp

            num_samples = 1000
            samples = posterior.sample_batched((num_samples,), x=x_val, show_progress_bars=False)
            ecp, _ = tarp.get_tarp_coverage(
                samples, theta_val, norm=True, bootstrap=True, num_bootstrap=200
            )

            tarp_val = torch.mean(torch.from_numpy(ecp[:, ecp.shape[1] // 2])).to(self.device)
            return float(abs(tarp_val - 0.5))
        else:
            raise ValueError(f"Metric '{metric_name}' not recognized for custom objective.")

    @staticmethod
    def _train_model(
        density_estimator: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        max_num_epochs: int = 5000,
        stop_after_epochs: int = 20,
        trial: Optional[optuna.Trial] = None,
    ) -> Tuple[nn.Module, float, Dict]:
        best_val_loss, epochs_since_improvement = float("Inf"), 0
        best_model_state_dict = None
        train_log, val_log = [], []
        for epoch in range(max_num_epochs):
            # Training phase
            density_estimator.train()
            epoch_train_loss = 0.0
            for theta_batch, x_batch in train_loader:
                optimizer.zero_grad()
                train_loss = density_estimator.loss(theta_batch, x_batch).mean()
                train_loss.backward()
                optimizer.step()
                epoch_train_loss += train_loss.item()
            train_log.append(epoch_train_loss / len(train_loader))

            # Validation phase
            density_estimator.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for theta_batch, x_batch in val_loader:
                    val_loss_sum += density_estimator.loss(theta_batch, x_batch).sum().item()
            current_val_loss = (
                val_loss_sum / len(val_loader.sampler)
                if len(val_loader.sampler) > 0
                else float("Inf")
            )
            val_log.append(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss, epochs_since_improvement = current_val_loss, 0
                best_model_state_dict = deepcopy(density_estimator.state_dict())
            else:
                epochs_since_improvement += 1

            if trial:
                trial.report(current_val_loss, epoch)
                if trial.should_prune():
                    logging.info(f"Trial pruned at epoch {epoch + 1}.")
                    raise optuna.TrialPruned()
            elif epochs_since_improvement >= stop_after_epochs:
                logging.info(f"Stopping early after {epoch + 1} epochs.")
                break

        if best_model_state_dict:
            density_estimator.load_state_dict(best_model_state_dict)

        summary = {
            "training_loss": train_log,
            "validation_loss": val_log,
            "best_validation_loss": [best_val_loss],
        }
        return density_estimator, best_val_loss, summary
