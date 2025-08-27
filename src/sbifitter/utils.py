"""Utility functions for SBIFitter."""

import io
import operator
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Union, TextIO
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from unyt import Angstrom, Jy, nJy, unyt_array, unyt_quantity


def load_grid_from_hdf5(
    hdf5_path: str,
    photometry_key: str = "Grid/Photometry",
    parameters_key: str = "Grid/Parameters",
    filter_codes_attr: str = "FilterCodes",
    parameters_attr: str = "ParameterNames",
    parameters_units_attr: str = "ParameterUnits",
    supp_key: str = "Grid/SupplementaryParameters",
    supp_attr: str = "SupplementaryParameterNames",
    supp_units_attr: str = "SupplementaryParameterUnits",
    phot_unit_attr: str = "PhotometryUnits",
    spectra_key: str = "Grid/Spectra",
) -> dict:
    """Load a grid from an HDF5 file.

    Parameters:
        hdf5_path: Path to the HDF5 file.
        photometry_key: Key for the photometry dataset in the HDF5 file.
        parameters_key: Key for the parameters dataset in the HDF5 file.
        filter_codes_attr: Attribute name for filter codes in the HDF5 file.
        parameters_attr: Attribute name for parameter names in the HDF5 file.
        supp_key: Key for supplementary parameters in the HDF5 file.
        supp_attr: Attribute name for supplementary parameter names in the HDF5 file.
        supp_units_attr: Attribute name for supplementary parameter units in HDF5 file.
        phot_unit_attr: Attribute name for photometry units in the HDF5 file.

    Returns:
        The loaded grid.
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Load the photometry and parameters from the HDF5 file

        parameters = f[parameters_key][:]

        filter_codes = f.attrs[filter_codes_attr]
        parameter_names = f.attrs[parameters_attr]

        # Load the photometry units
        photometry_units = f.attrs[phot_unit_attr]

        parameter_units = f.attrs.get(parameters_units_attr, None)

        output = {
            "parameters": parameters,
            "filter_codes": filter_codes,
            "parameter_names": parameter_names,
            "photometry_units": photometry_units,
            "parameter_units": parameter_units,
        }

        if photometry_key in f:
            photometry = f[photometry_key][:]
            output["photometry"] = photometry
        if spectra_key in f:
            spectra = f[spectra_key][:]
            output["spectra"] = spectra

        # Load supplementary parameters if available
        if supp_key in f:
            supplementary_parameters = f[supp_key][:]
            supplementary_parameter_names = f.attrs[supp_attr]
            supplementary_parameter_units = f.attrs[supp_units_attr]
            output["supplementary_parameters"] = supplementary_parameters
            output["supplementary_parameter_names"] = supplementary_parameter_names
            output["supplementary_parameter_units"] = supplementary_parameter_units

    return output


def calculate_min_max_wav_grid(filterset, max_redshift, min_redshift=0):
    """Calculate the minimum and maximum wavelengths for a given redshift."""
    # Get the filter limits
    filter_lims = filterset.get_non_zero_lam_lims()

    # Calculate the maximum wavelength for a given redshift
    max_wav = filter_lims[1] / (1 + min_redshift)

    # Calculate the minimum wavelength for a given redshift
    min_wav = filter_lims[0] / (1 + max_redshift)

    return min_wav, max_wav


def generate_constant_R(
    R=300,
    start=1 * Angstrom,
    end=9e5 * Angstrom,
    auto_start_stop=False,
    filterset=None,
    **kwargs,
):
    """Generate a constant R wavelength grid.

    Parameters:
        R: The resolution of the grid.
        start: The starting wavelength of the grid.
        end: The ending wavelength of the grid.
        auto_start_stop: If True, calculate start and end from the filterset.
        filterset: A filter set to calculate the start and end wavelengths.
        **kwargs: Additional keyword arguments for filterset calculations.

    Returns:
        A numpy array of wavelengths in Angstroms.
    """
    if auto_start_stop and filterset is not None:
        start, end = calculate_min_max_wav_grid(filterset, **kwargs)

    x = [start.to(Angstrom).value]

    while x[-1] < end.to(Angstrom).value:
        x.append(x[-1] * (1.0 + 0.5 / R))

    return np.array(x) * Angstrom


def list_parameters(distribution):
    """List parameters for scipy.stats.distribution.

    Parameters
        distribution: a string or scipy.stats distribution object.

    Returns:
        A list of distribution parameter strings.
    # from https://stackoverflow.com/questions/30453097/getting-the-parameter-names-of-scipy-stats-distributions
    """
    if isinstance(distribution, str):
        distribution = getattr(stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(",")]
    else:
        parameters = []
    if distribution.name in stats._discrete_distns._distn_names:
        parameters += ["loc"]
    elif distribution.name in stats._continuous_distns._distn_names:
        parameters += ["loc", "scale"]
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters


def rename_overlapping_parameters(lists_dict):
    """Check if N lists have any overlapping parameters and rename them if they do.

    Args:
        lists_dict: Dictionary where keys are list names and values are the lists

    Returns:
        Dictionary with renamed parameters where overlapping occurred
    """
    # Collect all parameters across all lists
    all_params = {}
    for list_name, params in lists_dict.items():
        for param in params:
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(list_name)

    # Build the result with renamed parameters where needed
    result = {}
    for list_name, params in lists_dict.items():
        result[list_name] = []
        for param in params:
            # If parameter appears in multiple lists, rename it
            if len(all_params[param]) > 1:
                result[list_name].append(f"{list_name}_{param}")
            else:
                result[list_name].append(param)

    return result


class FilterArithmeticParser:
    """Parser for filter arithmetic expressions.

    Parser for filter arithmetic expressions.
    Supports operations like:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Constants and coefficients

    Examples:
        "F356W"                    -> single filter
        "F356W + F444W"           -> filter addition
        "2 * F356W"               -> coefficient multiplication
        "(F356W + F444W) / 2"     -> average of filters
        "F356W - 0.5 * F444W"     -> weighted subtraction
    """

    def __init__(self):
        """Initialize the FilterArithmeticParser."""
        self.operators = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        # Regular expression pattern for tokenizing
        self.pattern = r"(\d*\.\d+|\d+|[A-Za-z]\d+[A-Za-z]+|\+|\-|\*|\/|\(|\))"

    def tokenize(self, expression: str) -> List[str]:
        """Convert string expression into list of tokens."""
        tokens = re.findall(self.pattern, expression)
        return [token.strip() for token in tokens if token.strip()]

    def is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def is_filter(self, token: str) -> bool:
        """Check if token is a filter name."""
        return bool(re.match(r"^[A-Za-z]\d+[A-Za-z]+$", token))

    def evaluate(
        self,
        tokens: List[str],
        filter_data: Dict[str, Union[float, np.ndarray]],
    ) -> Union[float, np.ndarray]:
        """Evaluate a list of tokens using provided filter data.

        Args:
            tokens: List of tokens from the expression
            filter_data: Dictionary mapping filter names to their values

        Returns:
            Result of the arithmetic operations
        """
        output_stack = []
        operator_stack = []

        precedence = {"+": 1, "-": 1, "*": 2, "/": 2}

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == "(":
                operator_stack.append(token)

            elif token == ")":
                while operator_stack and operator_stack[-1] != "(":
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.pop()  # Remove '('

            elif token in self.operators:
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and precedence.get(operator_stack[-1], 0) >= precedence[token]
                ):
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.append(token)

            else:  # Number or filter name
                if self.is_number(token):
                    value = float(token)
                elif self.is_filter(token):
                    if token not in filter_data:
                        raise ValueError(f"Filter {token} not found in provided data")
                    value = filter_data[token]
                else:
                    raise ValueError(f"Invalid token: {token}")
                output_stack.append(value)

            i += 1

        while operator_stack:
            self._apply_operator(operator_stack, output_stack, filter_data)

        if len(output_stack) != 1:
            raise ValueError(f"Invalid expression: {output_stack}")

        return output_stack[0]

    def _apply_operator(
        self,
        operator_stack: List[str],
        output_stack: List[Union[float, np.ndarray]],
        filter_data: Dict[str, Union[float, np.ndarray]],
    ) -> None:
        """Apply operator to the top two values in the output stack."""
        operator = operator_stack.pop()
        b = output_stack.pop()
        a = output_stack.pop()
        output_stack.append(self.operators[operator](a, b))

    def parse_and_evaluate(
        self, expression: str, filter_data: Dict[str, Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """Parse and evaluate a filter arithmetic expression.

        Args:
            expression: String containing the filter arithmetic expression
            filter_data: Dictionary mapping filter names to their values

        Returns:
            Result of evaluating the expression
        """
        tokens = self.tokenize(expression)
        return self.evaluate(tokens, filter_data)


def timeout_handler(signum, frame):
    """Handler for alarm signal."""
    raise TimeoutException


class TimeoutException(Exception):
    """Exception raised when a function times out."""

    pass


def create_sqlite_db(db_path: str):
    """Create a SQLite database at the specified path.

    Parameters:
        db_path: Path to the SQLite database file.
    """
    import sqlite3

    if not db_path.endswith(".db"):
        db_path += ".db"
    try:
        sqlite3.connect(db_path)
    except sqlite3.OperationalError as e:
        print("Failed to open database:", e)

    storage_name = "sqlite:///{}".format(db_path)

    print(storage_name)

    return storage_name

def create_database_universal(
    db_name: str,
    password: str = "",
    host: str = "localhost",
    user: str = "root",
    port: int = 31666,
    db_type = "mysql+pymysql",
    full_url: str = None
):
    """
    Create database for MySQL, PostgreSQL, or CockroachDB
    Returns the full connection URL for the created database.

    Either provide a full URL or the individual parameters.
    """
    import sqlalchemy
    from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
    from urllib.parse import urlparse, urlunparse
    
    assert db_type in ["mysql+pymysql", "postgresql+psycopg2", "cockroachdb"], (
        "db_type must be one of 'mysql+pymysql', 'postgresql+psycopg2', or 'cockroachdb'."
    )
    # Determine database type and prepare connection details
    if full_url is None:
        sqlalchemy_url = f":{db_type}//{user}:{password}@{host}:{port}/"
    else:
        parsed = urlparse(full_url)
        if full_url.startswith("mysql://"):
            db_type = "mysql"
            sqlalchemy_url = full_url.replace("mysql://", "mysql+pymysql://")
        elif full_url.startswith("postgres://") or full_url.startswith("postgresql://"):
            db_type = "postgresql"
            if full_url.startswith("postgres://"):
                sqlalchemy_url = full_url.replace("postgres://", "postgresql+psycopg2://")
            else:
                sqlalchemy_url = full_url.replace("postgresql://", "postgresql+psycopg2://")
        elif full_url.startswith("cockroachdb://"):
            db_type = "cockroachdb"
            sqlalchemy_url = full_url  # Keep cockroachdb:// scheme
        else:
            db_type = "unknown"
            sqlalchemy_url = full_url
    
    # Remove existing database name from URL to connect to system database
    parsed = urlparse(sqlalchemy_url)
    path_parts = parsed.path.strip('/').split('/') if parsed.path.strip('/') else []
    
    if db_type == "mysql":
        # Connect to mysql system database
        system_path = "/mysql"
        cd = "`"  # MySQL uses backticks
        create_sql = f"CREATE DATABASE IF NOT EXISTS {cd}{db_name}{cd}"
        isolation_level = "AUTOCOMMIT"
    elif db_type == "postgresql":
        # Connect to postgres system database
        system_path = "/postgres"
        cd = '"'  # PostgreSQL uses double quotes
        create_sql = f'CREATE DATABASE {cd}{db_name}{cd}'
        isolation_level = "AUTOCOMMIT"
    elif db_type == "cockroachdb":
        # CockroachDB uses defaultdb as system database
        system_path = "/defaultdb"
        cd = ""  # CockroachDB doesn't require quotes for simple names
        create_sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"
        db_name = db_name.lower()
        isolation_level = "AUTOCOMMIT"
    else:
        # Fallback - try without quotes
        system_path = "/"
        cd = ""
        create_sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"
        isolation_level = "AUTOCOMMIT"
    
    # Build system database URL
    system_url = parsed._replace(path=system_path)
    system_url = urlunparse(system_url)
    
    try:
        engine = sqlalchemy.create_engine(system_url, isolation_level=isolation_level)
        
        with engine.connect() as connection:
            if db_type == "postgresql":
                # PostgreSQL: Check if database exists first
                try:
                    result = connection.execute(
                        sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                        {"db_name": db_name}
                    )
                    if not result.fetchone():
                        connection.execute(sqlalchemy.text(create_sql))
                        print(f"Database {db_name} created.")
                    else:
                        print(f"Database {db_name} already exists.")
                except ProgrammingError as e:
                    if "already exists" not in str(e).lower():
                        raise
                    print(f"Database {db_name} already exists.")
            else:
                # MySQL and CockroachDB: Use IF NOT EXISTS
                connection.execute(sqlalchemy.text(create_sql))
                print(f"Database {db_name} created or already exists.")
        
        engine.dispose()
        
    except SQLAlchemyError as e:
        print(f"Failed to create database {db_name}:", e)
        # Continue anyway - database might already exist
    
    # Build final database URL
    if full_url is None:
        final_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    else:
        # Replace system database with target database in original URL
        parsed = urlparse(sqlalchemy_url)
        if db_type == "cockroachdb" and 'defaultdb' in sqlalchemy_url:
            final_url = sqlalchemy_url.replace('defaultdb', db_name)
        else:
            final_url = parsed._replace(path=f"/{db_name}")
            final_url = urlunparse(final_url)
        
        # Convert back to original scheme format if needed
        if full_url.startswith("mysql://") and final_url.startswith("mysql+pymysql://"):
            final_url = final_url.replace("mysql+pymysql://", "mysql://")
        elif full_url.startswith("postgres://") and final_url.startswith("postgresql+psycopg2://"):
            final_url = final_url.replace("postgresql+psycopg2://", "postgres://")
        # CockroachDB keeps its original scheme
    
    return final_url


def f_jy_to_asinh(
    f_jy: unyt_array,
    f_b: unyt_array = 5 * nJy,  # 29.6 AB mag
) -> np.ndarray:
    """Convert flux in Jy to asinh magnitude.

    Parameters:
        f_jy: Flux in Jy.
        f_b:  Softening parameter (transition point for the asinh scale).

    Returns:
        Magnitude in asinh scale.
    """
    f_jy = f_jy.to(Jy)
    f_b = f_b.to(Jy)

    if f_b.ndim == 0:
        f_b = np.full_like(f_jy, f_b.value, dtype=f_jy.dtype)
    elif f_b.ndim == 1 and f_jy.ndim == 2:
        assert f_b.shape[0] == f_jy.shape[0], "Flux softening must match the number of filters."
        f_b = np.tile(f_b, (f_jy.shape[1], 1)).T

    else:
        assert f_b.shape == f_jy.shape, "Flux and flux softening must have the same shape."

    asinh = -2.5 * np.log10(np.e) * (np.arcsinh(f_jy / (2 * f_b)) + np.log(f_b / (3631 * Jy)))
    return asinh


def f_jy_err_to_asinh(
    f_jy: unyt_array,
    f_jy_err: unyt_array,
    f_b: unyt_array = 5 * nJy,  # 29.6 AB mag
) -> np.ndarray:
    """Convert flux error in Jy to asinh magnitude error.

    Parameters:
        f_jy: Flux in Jy.
        f_jy_err: Flux error in Jy.
        f_b: Softening parameter (transition point for the asinh scale).

    Returns:
        Magnitude error in asinh scale.
    """
    f_jy = f_jy.to(Jy).value
    f_jy_err = f_jy_err.to(Jy).value

    assert f_jy.shape == f_jy_err.shape, "Flux and flux error must have the same shape."
    if f_b.ndim == 0:
        f_b = unyt_array(np.full_like(f_jy, f_b.value, dtype=f_jy.dtype), units=f_b.units)
    elif f_b.ndim == 1 and f_jy.ndim == 2:
        assert f_b.shape[0] == f_jy.shape[0], "Flux softening must match the number of filters."
        f_b = np.tile(f_b, (f_jy.shape[1], 1)).T
    else:
        assert f_b.shape == f_jy.shape, "Flux and flux error must have the same shape."

    f_b = f_b.to(Jy).value
    return 2.5 * np.log10(np.e) * f_jy_err / np.sqrt(f_jy**2 + (2 * f_b) ** 2)


def save_emission_model(model):
    """Save the fixed parameters of the emission model.

    Parameters:
        model: The emission model object.

    Returns:
        A dictionary containing fixed parameters, dust attenuation,
        and dust emission model information.
    """
    fixed_params = model.fixed_parameters

    if fixed_params is None:
        fixed_params = {}

    for k, m in model._models.items():
        for i, j in m.fixed_parameters.items():
            if i not in fixed_params.keys():
                fixed_params[i] = j
            else:
                if isinstance(fixed_params[i], str) and not isinstance(j, str):
                    fixed_params[i] = j

    dust_attenuation_keys = {}
    if "attenuated" in model._transformation.keys():
        dust_law = model._transformation["attenuated"][1]
        dust_attenuation_keys.update(dust_law.__dict__)
        dust_attenuation_keys.pop("description")
        dust_attenuation_keys.pop("_required_params")
        dust_law = type(dust_law).__name__

    else:
        dust_law = None

    dust_emission_keys = {}

    if "dust_emission" in model._models:
        dust_em = model._models["dust_emission"].generator
        dust_emission_keys.update(dust_em.__dict__)
        dust_emission_model = type(dust_em).__name__
    else:
        dust_emission_model = None

    fixed_param_units = []
    for k, v in fixed_params.items():
        if hasattr(v, "units"):
            fixed_param_units.append(str(v.units))
            fixed_params[k] = v.value
        else:
            fixed_param_units.append("")

    fixed_parameter_keys = list(fixed_params.keys())
    fixed_parameter_values = list(fixed_params.values())
    # if any strings in fixed_parameter_values, convert all to string
    if any(isinstance(v, str) for v in fixed_parameter_values):
        fixed_parameter_values = [str(v) for v in fixed_parameter_values]

    dust_attenuation_units = []
    for k, v in dust_attenuation_keys.items():
        if hasattr(v, "units"):
            dust_attenuation_units.append(str(v.units))
            dust_attenuation_keys[k] = v.value
        else:
            dust_attenuation_units.append("")

    dust_attenuation_values = list(dust_attenuation_keys.values())
    dust_attenuation_keys = list(dust_attenuation_keys.keys())

    dust_emission_units = []
    for k, v in dust_emission_keys.items():
        if hasattr(v, "units"):
            dust_emission_units.append(str(v.units))
            dust_emission_keys[k] = v.value
        else:
            dust_emission_units.append("")

    dust_emission_values = list(dust_emission_keys.values())
    dust_emission_keys = list(dust_emission_keys.keys())

    return {
        "fixed_parameter_keys": fixed_parameter_keys,
        "fixed_parameter_values": fixed_parameter_values,
        "fixed_parameter_units": fixed_param_units,
        "dust_law": dust_law,
        "dust_attenuation_keys": dust_attenuation_keys,
        "dust_attenuation_values": dust_attenuation_values,
        "dust_attenuation_units": dust_attenuation_units,
        "dust_emission": dust_emission_model,
        "dust_emission_keys": dust_emission_keys,
        "dust_emission_values": dust_emission_values,
        "dust_emission_units": dust_emission_units,
    }


class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler that handles specific Torch storage loading."""

    def find_class(self, module, name):
        """Find class in the specified module."""
        import torch

        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def check_log_scaling(arr: Union[unyt_array, unyt_quantity]):
    """Check if the input array has dimensions that scale with logarithmic normalization."""
    assert isinstance(arr, unyt_array) or isinstance(arr, unyt_quantity), (
        "Input must be a unyt_array or unyt_quantity, got {}".format(type(arr))
    )

    if isinstance(arr, unyt_quantity):
        arr = 1.0 * arr  # Ensure it is a unyt_array

    if not isinstance(arr, unyt_array):
        return False

    # Check if the unit is dimensionless
    if arr.units.is_dimensionless and "log" in str(arr.units):
        return True


def check_scaling(arr: Union[unyt_array, unyt_quantity]):
    """Check if the input array has dimensions that scale with normalization."""
    assert isinstance(arr, unyt_array) or isinstance(arr, unyt_quantity), (
        "Input must be a unyt_array or unyt_quantity, got {}".format(type(arr))
    )

    from unyt.dimensions import (
        energy,
        flux,
        luminance,
        luminous_flux,
        mass,
        power,
        specific_flux,
        time,
    )

    if isinstance(arr, unyt_quantity):
        arr = 1.0 * arr  # Ensure it is a unyt_array

    if not isinstance(arr, unyt_array):
        return False

    # We want to check how the unit scales.
    # If it is a flux, flux density, luminosity, mass etc or other parameters
    # which scale with normalization, we want to return True
    # if it is a distance, time, dimensionless parameter, we want to return False
    if arr.units.is_dimensionless:
        return False

    if arr.units.dimensions in (
        energy,
        power,
        flux,
        specific_flux,
        luminance,
        mass,
        luminous_flux,
        mass / time,
    ):
        return True

    return False


def detect_outliers(
    base_distribution,
    observations,
    method="mahalanobis",
    contamination=0.1,
    n_neighbors=20,
    threshold=None,
    confidence=0.95,
    n_components=None,
    plot=True,
    **kwargs,
):
    """Detect outliers in multivariate data using various methods.

    Parameters:
    -----------
    base_distribution : array-like, shape (n_samples, n_features)
        Reference distribution data
    observations : array-like, shape (n_obs, n_features)
        Observations to test for outliers
    method : str, default='mahalanobis'
        Method to use: 'mahalanobis', 'robust_mahalanobis', 'lof', 'isolation_forest',
        'one_class_svm', 'pca', 'hotelling_t2', 'kde'
    contamination : float, default=0.1
        Expected proportion of outliers (for applicable methods)
    n_neighbors : int, default=20
        Number of neighbors for LOF
    threshold : float, optional
        Manual threshold for outlier detection
    confidence : float, default=0.95
        Confidence level for statistical tests
    n_components : int, optional
        Number of components for PCA (if None, uses all)
    plot : bool, default=True
        Whether to plot results (only applicable for some methods)
    **kwargs : dict
        Additional parameters for specific methods

    Returns:
    --------
    dict : Dictionary containing:
        - 'outlier_mask': Boolean array indicating outliers
        - 'scores': Outlier scores
        - 'threshold_used': Threshold value used
        - 'method_info': Additional method-specific information
    """
    from sklearn.covariance import EllipticEnvelope
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    base_distribution = np.asarray(base_distribution)
    observations = np.asarray(observations)

    # Ensure same number of features
    if base_distribution.shape[1] != observations.shape[1]:
        raise ValueError("Base distribution and observations must have same number of features")

    n_features = base_distribution.shape[1]
    n_obs = observations.shape[0]

    results = {
        "outlier_mask": np.zeros(n_obs, dtype=bool),
        "scores": np.zeros(n_obs),
        "threshold_used": None,
        "method_info": {},
    }

    if method == "mahalanobis":
        # Standard Mahalanobis distance
        mean = np.mean(base_distribution, axis=0)
        cov = np.cov(base_distribution.T)

        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        # Calculate Mahalanobis distances
        diff = observations - mean
        mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        # Threshold based on chi-squared distribution
        if threshold is None:
            threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))

        results["scores"] = mahal_dist
        results["outlier_mask"] = mahal_dist > threshold
        results["threshold_used"] = threshold
        results["method_info"] = {"mean": mean, "covariance": cov}

    elif method == "robust_mahalanobis":
        # Robust Mahalanobis using Minimum Covariance Determinant
        robust_cov = EllipticEnvelope(contamination=contamination)
        robust_cov.fit(base_distribution)

        # Get robust estimates
        mean = robust_cov.location_
        cov = robust_cov.covariance_

        # Calculate robust Mahalanobis distances
        diff = observations - mean
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        if threshold is None:
            threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))

        results["scores"] = mahal_dist
        results["outlier_mask"] = mahal_dist > threshold
        results["threshold_used"] = threshold
        results["method_info"] = {"robust_mean": mean, "robust_covariance": cov}

    elif method == "lof":
        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
        lof.fit(base_distribution)

        # Predict outliers (-1 for outliers, 1 for inliers)
        predictions = lof.predict(observations)
        scores = lof.decision_function(observations)

        results["scores"] = -scores  # Make positive scores indicate outliers
        results["outlier_mask"] = predictions == -1
        results["threshold_used"] = 0  # LOF uses 0 as threshold
        results["method_info"] = {"n_neighbors": n_neighbors}

    elif method == "isolation_forest":
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(base_distribution)

        predictions = iso_forest.predict(observations)
        scores = iso_forest.decision_function(observations)

        results["scores"] = -scores  # Make positive scores indicate outliers
        results["outlier_mask"] = predictions == -1
        results["threshold_used"] = 0
        results["method_info"] = {"contamination": contamination}

    elif method == "one_class_svm":
        # One-Class SVM
        gamma = kwargs.get("gamma", "scale")
        nu = kwargs.get("nu", contamination)

        svm = OneClassSVM(gamma=gamma, nu=nu)
        svm.fit(base_distribution)

        predictions = svm.predict(observations)
        scores = svm.decision_function(observations)

        results["scores"] = -scores
        results["outlier_mask"] = predictions == -1
        results["threshold_used"] = 0
        results["method_info"] = {"gamma": gamma, "nu": nu}

    elif method == "pca":
        # PCA-based outlier detection
        if n_components is None:
            n_components = min(n_features, base_distribution.shape[0] - 1)

        pca = PCA(n_components=n_components)
        pca.fit(base_distribution)

        # Transform observations to PCA space
        obs_transformed = pca.transform(observations)

        # Reconstruct and calculate reconstruction error
        obs_reconstructed = pca.inverse_transform(obs_transformed)
        reconstruction_error = np.sum((observations - obs_reconstructed) ** 2, axis=1)

        if threshold is None:
            # Use percentile of reconstruction errors from base distribution
            base_transformed = pca.transform(base_distribution)
            base_reconstructed = pca.inverse_transform(base_transformed)
            base_errors = np.sum((base_distribution - base_reconstructed) ** 2, axis=1)
            threshold = np.percentile(base_errors, confidence * 100)

        results["scores"] = reconstruction_error
        results["outlier_mask"] = reconstruction_error > threshold
        results["threshold_used"] = threshold
        results["method_info"] = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

    elif method == "hotelling_t2":
        # Hotelling's T² test for multivariate normality
        mean = np.mean(base_distribution, axis=0)
        cov = np.cov(base_distribution.T)
        n_base = base_distribution.shape[0]

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        # Calculate T² statistics
        diff = observations - mean
        t2_stats = np.sum(diff @ inv_cov * diff, axis=1) * n_base

        # Convert to F-distribution
        f_stats = t2_stats * (n_base - n_features) / ((n_base - 1) * n_features)

        if threshold is None:
            threshold = stats.f.ppf(confidence, n_features, n_base - n_features)

        results["scores"] = f_stats
        results["outlier_mask"] = f_stats > threshold
        results["threshold_used"] = threshold
        results["method_info"] = {
            "n_base_samples": n_base,
            "degrees_of_freedom": (n_features, n_base - n_features),
        }

    elif method == "kde":
        # Kernel Density Estimation
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(base_distribution.T)
            densities = kde(observations.T)

            # Use low density as outlier indicator
            base_densities = kde(base_distribution.T)
            if threshold is None:
                threshold = np.percentile(base_densities, (1 - confidence) * 100)

            results["scores"] = -np.log(densities + 1e-10)  # Use negative log density
            results["outlier_mask"] = densities < threshold
            results["threshold_used"] = threshold
            results["method_info"] = {"kde_bandwidth": kde.factor}
        except Exception as e:
            print(f"KDE failed: {e}. Using fallback method.")
            # Fallback to Mahalanobis
            return detect_outliers(
                base_distribution,
                observations,
                method="mahalanobis",
                confidence=confidence,
                threshold=threshold,
            )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Display results
    if plot:
        _display_results(base_distribution, observations, results, method)

    return results


def _display_results(base_distribution, observations, results, method):
    """Display visualization of outlier detection results."""
    n_features = base_distribution.shape[1]
    n_outliers = np.sum(results["outlier_mask"])
    n_total = len(observations)

    print(f"\n=== Outlier Detection Results ({method.upper()}) ===")
    print(f"Total observations: {n_total}")
    print(f"Outliers detected: {n_outliers} ({n_outliers / n_total:.1%})")
    print(f"Threshold used: {results['threshold_used']:.4f}")

    if "method_info" in results and results["method_info"]:
        print("\nMethod-specific information:")
        for key, value in results["method_info"].items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1 and len(value) <= 5:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: array of shape {value.shape}")
            else:
                print(f"  {key}: {value}")

    # Create visualization for 2D data
    if n_features == 2:
        plt.figure(figsize=(12, 5))

        # Plot 1: Data distribution
        plt.subplot(1, 2, 1)
        plt.scatter(
            base_distribution[:, 0],
            base_distribution[:, 1],
            alpha=0.6,
            label="Base Distribution",
            c="lightblue",
            s=20,
        )

        inliers = observations[~results["outlier_mask"]]
        outliers = observations[results["outlier_mask"]]

        if len(inliers) > 0:
            plt.scatter(inliers[:, 0], inliers[:, 1], c="green", label="Inliers", s=50, alpha=0.8)
        if len(outliers) > 0:
            plt.scatter(outliers[:, 0], outliers[:, 1], c="red", label="Outliers", s=50, alpha=0.8)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"Outlier Detection - {method.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Outlier scores
        plt.subplot(1, 2, 2)
        plt.hist(results["scores"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        if results["threshold_used"] is not None:
            plt.axvline(
                results["threshold_used"],
                color="red",
                linestyle="--",
                label=f"Threshold: {results['threshold_used']:.3f}",
            )
        plt.xlabel("Outlier Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Outlier Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    elif n_features > 2:
        from sklearn.decomposition import PCA

        # For higher dimensional data, show score distribution
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(results["scores"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        if results["threshold_used"] is not None:
            plt.axvline(
                results["threshold_used"],
                color="red",
                linestyle="--",
                label=f"Threshold: {results['threshold_used']:.3f}",
            )
        plt.xlabel("Outlier Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Outlier Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Show first two principal components
        plt.subplot(1, 2, 2)
        pca = PCA(n_components=2)
        base_pca = pca.fit_transform(base_distribution)
        obs_pca = pca.transform(observations)

        plt.scatter(
            base_pca[:, 0],
            base_pca[:, 1],
            alpha=0.6,
            label="Base Distribution",
            c="lightblue",
            s=20,
        )

        inliers_pca = obs_pca[~results["outlier_mask"]]
        outliers_pca = obs_pca[results["outlier_mask"]]

        if len(inliers_pca) > 0:
            plt.scatter(
                inliers_pca[:, 0], inliers_pca[:, 1], c="green", label="Inliers", s=50, alpha=0.8
            )
        if len(outliers_pca) > 0:
            plt.scatter(
                outliers_pca[:, 0], outliers_pca[:, 1], c="red", label="Outliers", s=50, alpha=0.8
            )

        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title(f"PCA View - {method.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Print outlier indices
    outlier_indices = np.where(results["outlier_mask"])[0]
    if len(outlier_indices) > 0:
        print(
            f"\nOutlier indices: {outlier_indices[:10]}{'...' if len(outlier_indices) > 10 else ''}"
        )


def analyze_feature_contributions(
    base_distribution,
    observations,
    method="mahalanobis",
    feature_names=None,
    contamination=0.1,
    confidence=0.95,
):
    """Analyze which features contribute most to outlier detection in distance-based methods.

    Parameters:
    -----------
    base_distribution : array-like, shape (n_samples, n_features)
        Reference distribution data
    observations : array-like, shape (n_obs, n_features)
        Observations to analyze
    method : str, default='mahalanobis'
        Method to use: 'mahalanobis', 'robust_mahalanobis', or 'standardized_euclidean'
    feature_names : list, optional
        Names of features for plotting
    contamination : float, default=0.1
        Expected proportion of outliers (for robust methods)
    confidence : float, default=0.95
        Confidence level for thresholds

    Returns:
    --------
    dict : Dictionary containing feature contribution analysis
    """
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler

    base_distribution = np.asarray(base_distribution)
    observations = np.asarray(observations)

    n_features = base_distribution.shape[1]
    n_obs = observations.shape[0]

    if feature_names is None:
        feature_names = [f"Feature_{i + 1}" for i in range(n_features)]

    results = {
        "feature_names": feature_names,
        "method": method,
        "feature_contributions": np.zeros((n_obs, n_features)),
        "total_distances": np.zeros(n_obs),
        "feature_importance": np.zeros(n_features),
        "outlier_mask": np.zeros(n_obs, dtype=bool),
    }

    if method == "mahalanobis":
        mean = np.mean(base_distribution, axis=0)
        cov = np.cov(base_distribution.T)

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        # Calculate differences
        diff = observations - mean

        # Feature contributions to squared Mahalanobis distance
        # For each observation, contribution of feature i is: diff_i * sum_j(inv_cov[i,j] * diff_j)
        for i in range(n_obs):
            for j in range(n_features):
                results["feature_contributions"][i, j] = diff[i, j] * np.dot(
                    inv_cov[j, :], diff[i, :]
                )

        # Total squared distances
        squared_distances = np.sum(results["feature_contributions"], axis=1)
        results["total_distances"] = np.sqrt(squared_distances)

        # Threshold
        threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))
        results["outlier_mask"] = results["total_distances"] > threshold

        results["mean"] = mean
        results["covariance"] = cov
        results["inv_covariance"] = inv_cov

    elif method == "robust_mahalanobis":
        robust_cov = EllipticEnvelope(contamination=contamination)
        robust_cov.fit(base_distribution)

        mean = robust_cov.location_
        cov = robust_cov.covariance_

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        diff = observations - mean

        # Feature contributions
        for i in range(n_obs):
            for j in range(n_features):
                results["feature_contributions"][i, j] = diff[i, j] * np.dot(
                    inv_cov[j, :], diff[i, :]
                )

        squared_distances = np.sum(results["feature_contributions"], axis=1)
        results["total_distances"] = np.sqrt(squared_distances)

        threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))
        results["outlier_mask"] = results["total_distances"] > threshold

        results["robust_mean"] = mean
        results["robust_covariance"] = cov
        results["inv_covariance"] = inv_cov

    elif method == "standardized_euclidean":
        # Standardize features independently
        scaler = StandardScaler()
        obs_scaled = scaler.transform(observations)

        # Each feature contributes independently
        results["feature_contributions"] = obs_scaled**2
        results["total_distances"] = np.sqrt(np.sum(results["feature_contributions"], axis=1))

        # Threshold based on chi-squared (since standardized features are approximately normal)
        threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))
        results["outlier_mask"] = results["total_distances"] > threshold

        results["feature_means"] = scaler.mean_
        results["feature_stds"] = scaler.scale_

    # Calculate feature importance (average absolute contribution across all observations)
    results["feature_importance"] = np.mean(np.abs(results["feature_contributions"]), axis=0)

    # Normalize feature importance to sum to 1
    results["feature_importance_normalized"] = results["feature_importance"] / np.sum(
        results["feature_importance"]
    )

    # Display results
    _display_feature_analysis(results)

    return results


def _display_feature_analysis(results):
    """Display comprehensive feature contribution analysis."""
    feature_names = results["feature_names"]
    n_features = len(feature_names)
    n_obs = len(results["total_distances"])
    outlier_mask = results["outlier_mask"]

    print(f"\n=== FEATURE CONTRIBUTION ANALYSIS ({results['method'].upper()}) ===")
    print(f"Total observations: {n_obs}")
    print(f"Outliers detected: {np.sum(outlier_mask)} ({np.sum(outlier_mask) / n_obs:.1%})")
    print(f"Number of features: {n_features}")

    # Feature importance ranking
    importance_order = np.argsort(results["feature_importance_normalized"])[::-1]

    print("\nFEATURE IMPORTANCE RANKING:")
    print("-" * 50)
    for i, feat_idx in enumerate(importance_order):
        print(
            f"{i + 1:2d}. {feature_names[feat_idx]:15s}: {results['feature_importance_normalized'][feat_idx]:.3f} "  # noqa E501
            f"({results['feature_importance_normalized'][feat_idx] * 100:.1f}%)"
        )

    # Outlier-specific analysis
    if np.sum(outlier_mask) > 0:
        print("\nOUTLIER-SPECIFIC FEATURE CONTRIBUTIONS:")
        print("-" * 50)

        outlier_contributions = results["feature_contributions"][outlier_mask]
        avg_outlier_contrib = np.mean(np.abs(outlier_contributions), axis=0)
        outlier_importance = avg_outlier_contrib / np.sum(avg_outlier_contrib)

        outlier_order = np.argsort(outlier_importance)[::-1]

        for i, feat_idx in enumerate(outlier_order[:5]):  # Top 5
            print(
                f"{i + 1:2d}. {feature_names[feat_idx]:15s}: {outlier_importance[feat_idx]:.3f} "
                f"({outlier_importance[feat_idx] * 100:.1f}%)"
            )

    # Create comprehensive visualization
    plt.figure(figsize=(20, 15))

    # 1. Feature importance bar plot
    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(range(n_features), results["feature_importance_normalized"])
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Normalized Importance")
    ax1.set_title("Feature Importance in Outlier Detection")
    ax1.set_xticks(range(n_features))
    ax1.set_xticklabels(feature_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Color bars by importance
    colors = plt.cm.viridis(
        results["feature_importance_normalized"] / np.max(results["feature_importance_normalized"])
    )
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 2. Contribution heatmap for outliers
    ax2 = plt.subplot(3, 3, 2)
    if np.sum(outlier_mask) > 0:
        outlier_contrib = results["feature_contributions"][outlier_mask]
        im = ax2.imshow(outlier_contrib.T, aspect="auto", cmap="RdBu_r")
        ax2.set_xlabel("Outlier Index")
        ax2.set_ylabel("Features")
        ax2.set_title("Feature Contributions for Outliers")
        ax2.set_yticks(range(n_features))
        ax2.set_yticklabels(feature_names)
        plt.colorbar(im, ax=ax2, label="Contribution")
    else:
        ax2.text(
            0.5, 0.5, "No outliers detected", ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_title("Feature Contributions for Outliers")

    # 3. Distance distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(results["total_distances"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax3.axvline(
        np.mean(results["total_distances"]),
        color="red",
        linestyle="--",
        label="Mean Distance",
        alpha=0.8,
    )
    if np.sum(outlier_mask) > 0:
        ax3.axvline(
            np.min(results["total_distances"][outlier_mask]),
            color="orange",
            linestyle="--",
            label="Min Outlier Distance",
            alpha=0.8,
        )
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Distances")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Feature contribution boxplots
    ax4 = plt.subplot(3, 3, 4)
    contrib_data = []
    labels = []
    for i, name in enumerate(feature_names):
        contrib_data.append(results["feature_contributions"][:, i])
        labels.append(name)

    ax4.boxplot(contrib_data, labels=labels)
    ax4.set_xlabel("Features")
    ax4.set_ylabel("Contribution")
    ax4.set_title("Distribution of Feature Contributions")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    # 5. Scatter plot of top 2 contributing features
    ax5 = plt.subplot(3, 3, 5)
    top_2_features = importance_order[:2]

    if results["method"] in ["mahalanobis", "robust_mahalanobis"]:
        # Use original feature values
        base_vals = results.get("base_distribution", np.zeros((1, n_features)))
        obs_vals = results.get("observations", np.zeros((n_obs, n_features)))

        # If we don't have the original data, reconstruct from contributions
        if base_vals.shape[0] == 1:
            # Use mean values as approximation
            # if "mean" in results:
            #    base_mean = results["mean"]
            # elif "robust_mean" in results:
            #    base_mean = results["robust_mean"]
            # else:
            #    base_mean = np.zeros(n_features)

            # Approximate observation values (this is a simplification)
            ax5.scatter(
                results["feature_contributions"][:, top_2_features[0]],
                results["feature_contributions"][:, top_2_features[1]],
                c=results["total_distances"],
                cmap="viridis",
                alpha=0.7,
            )
            ax5.set_xlabel(f"{feature_names[top_2_features[0]]} (Contribution)")
            ax5.set_ylabel(f"{feature_names[top_2_features[1]]} (Contribution)")
        else:
            ax5.scatter(
                obs_vals[:, top_2_features[0]],
                obs_vals[:, top_2_features[1]],
                c=results["total_distances"],
                cmap="viridis",
                alpha=0.7,
            )
            ax5.set_xlabel(f"{feature_names[top_2_features[0]]} (Value)")
            ax5.set_ylabel(f"{feature_names[top_2_features[1]]} (Value)")
    else:
        ax5.scatter(
            results["feature_contributions"][:, top_2_features[0]],
            results["feature_contributions"][:, top_2_features[1]],
            c=results["total_distances"],
            cmap="viridis",
            alpha=0.7,
        )
        ax5.set_xlabel(f"{feature_names[top_2_features[0]]} (Contribution)")
        ax5.set_ylabel(f"{feature_names[top_2_features[1]]} (Contribution)")

    ax5.set_title("Top 2 Contributing Features")
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax5, label="Distance")

    # 6. Feature correlation with distance
    ax6 = plt.subplot(3, 3, 6)
    correlations = []
    for i in range(n_features):
        corr = np.corrcoef(
            np.abs(results["feature_contributions"][:, i]), results["total_distances"]
        )[0, 1]
        correlations.append(corr)

    bars = ax6.bar(range(n_features), correlations)
    ax6.set_xlabel("Features")
    ax6.set_ylabel("Correlation with Distance")
    ax6.set_title("Feature Correlation with Total Distance")
    ax6.set_xticks(range(n_features))
    ax6.set_xticklabels(feature_names, rotation=45, ha="right")
    ax6.grid(True, alpha=0.3)

    # Color bars by correlation
    colors = plt.cm.RdYlBu_r((np.array(correlations) + 1) / 2)
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 7. Cumulative feature importance
    ax7 = plt.subplot(3, 3, 7)
    sorted_importance = np.sort(results["feature_importance_normalized"])[::-1]
    cumulative_importance = np.cumsum(sorted_importance)

    ax7.plot(range(1, n_features + 1), cumulative_importance, "o-", linewidth=2, markersize=6)
    ax7.axhline(0.8, color="red", linestyle="--", alpha=0.7, label="80% threshold")
    ax7.axhline(0.9, color="orange", linestyle="--", alpha=0.7, label="90% threshold")
    ax7.set_xlabel("Number of Top Features")
    ax7.set_ylabel("Cumulative Importance")
    ax7.set_title("Cumulative Feature Importance")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Individual feature distributions for outliers vs inliers
    ax8 = plt.subplot(3, 3, 8)
    top_feature_idx = importance_order[0]

    if np.sum(outlier_mask) > 0 and np.sum(~outlier_mask) > 0:
        outlier_contrib = results["feature_contributions"][outlier_mask, top_feature_idx]
        inlier_contrib = results["feature_contributions"][~outlier_mask, top_feature_idx]

        ax8.hist(inlier_contrib, bins=20, alpha=0.7, label="Inliers", color="blue", density=True)
        ax8.hist(outlier_contrib, bins=20, alpha=0.7, label="Outliers", color="red", density=True)
        ax8.set_xlabel(f"{feature_names[top_feature_idx]} Contribution")
        ax8.set_ylabel("Density")
        ax8.set_title("Distribution of Top Feature Contributions")
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(
            0.5,
            0.5,
            "Insufficient data for comparison",
            ha="center",
            va="center",
            transform=ax8.transAxes,
        )
        ax8.set_title("Feature Distribution Comparison")

    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("tight")
    ax9.axis("off")

    # Create summary table
    summary_data = []
    for i, feat_idx in enumerate(importance_order[:5]):
        summary_data.append(
            [
                feature_names[feat_idx],
                f"{results['feature_importance_normalized'][feat_idx]:.3f}",
                f"{correlations[feat_idx]:.3f}",
                f"{np.mean(results['feature_contributions'][:, feat_idx]):.3f}",
                f"{np.std(results['feature_contributions'][:, feat_idx]):.3f}",
            ]
        )

    table = ax9.table(
        cellText=summary_data,
        colLabels=["Feature", "Importance", "Correlation", "Mean Contrib", "Std Contrib"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title("Top 5 Features Summary")

    plt.tight_layout()
    plt.show()

    # Print specific insights
    print("\nKEY INSIGHTS:")
    print("-" * 50)

    # Most important feature
    top_feature = importance_order[0]
    print(
        f"• Most important feature: {feature_names[top_feature]} "
        f"({results['feature_importance_normalized'][top_feature] * 100:.1f}% of total importance)"
    )

    # Feature concentration
    top_3_importance = np.sum(results["feature_importance_normalized"][importance_order[:3]])
    print(f"• Top 3 features account for {top_3_importance * 100:.1f}% of outlier detection")

    # Feature with highest correlation
    max_corr_idx = np.argmax(correlations)
    print(
        f"• Feature most correlated with distance: {feature_names[max_corr_idx]} "
        f"(correlation: {correlations[max_corr_idx]:.3f})"
    )

    if np.sum(outlier_mask) > 0:
        # Feature that drives outliers most
        outlier_contrib = results["feature_contributions"][outlier_mask]
        avg_outlier_contrib = np.mean(np.abs(outlier_contrib), axis=0)
        outlier_driver = np.argmax(avg_outlier_contrib)
        print(f"• Feature driving outliers most: {feature_names[outlier_driver]}")


def compare_methods_feature_importance(base_distribution, observations, feature_names=None):
    """Compare feature importance across different distance-based methods."""
    methods = ["mahalanobis", "robust_mahalanobis", "standardized_euclidean"]

    if feature_names is None:
        feature_names = [f"Feature_{i + 1}" for i in range(base_distribution.shape[1])]

    results = {}

    print("COMPARING FEATURE IMPORTANCE ACROSS METHODS")
    print("=" * 60)

    for method in methods:
        print(f"\nAnalyzing {method}...")
        results[method] = analyze_feature_contributions(
            base_distribution, observations, method=method, feature_names=feature_names
        )

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, method in enumerate(methods):
        importance = results[method]["feature_importance_normalized"]
        bars = axes[i].bar(range(len(feature_names)), importance)
        axes[i].set_title(f"{method.replace('_', ' ').title()}")
        axes[i].set_xlabel("Features")
        axes[i].set_ylabel("Normalized Importance")
        axes[i].set_xticks(range(len(feature_names)))
        axes[i].set_xticklabels(feature_names, rotation=45, ha="right")
        axes[i].grid(True, alpha=0.3)

        # Color bars
        colors = plt.cm.viridis(importance / np.max(importance))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    plt.tight_layout()
    plt.show()

    return results


def optimize_sfh_xlimit(ax, mass_threshold=0.001, buffer_fraction=0.2):
    """Stolen from EXPANSE.

    Optimizes the x-axis limits of a matplotlib plot containing SFR histories
    to focus on periods after each galaxy has formed a certain fraction of its final mass.
    Calculates cumulative mass from SFR data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object containing the SFR plots (SFR/yr vs time)
    mass_threshold : float, optional
        Fraction of final stellar mass to use as threshold (default: 0.01 for 1%)
    buffer_fraction : float, optional
        Fraction of the active time range to add as buffer (default: 0.1)

    Returns:
    --------
    float
        The optimal maximum x value for the plot
    """
    # Get all lines from the plot
    lines = ax.get_lines()
    if not lines:
        raise ValueError("No lines found in the plot")

    # Initialize variables to track the earliest time reaching mass threshold
    earliest_activity = 0

    # Check each line
    for line in lines:
        # Get the x and y data
        xdata = line.get_xdata()
        ydata = line.get_ydata()  # This is SFR/yr

        # Calculate time intervals (assuming uniform spacing)
        dt = np.abs(xdata[1] - xdata[0])

        # Calculate cumulative mass formed
        # Integrate SFR from observation time (x=0) backwards
        # Remember: x-axis is negative lookback time, so we need to flip the integration
        cumulative_mass = np.cumsum(ydata[::-1] * dt)[::-1]

        # Normalize by total mass formed
        total_mass = cumulative_mass[0]  # Mass at observation time
        normalized_mass = cumulative_mass / total_mass

        # Find indices where normalized mass exceeds threshold
        active_indices = np.where(normalized_mass >= mass_threshold)[0]

        if len(active_indices) > 0:
            # Find the earliest time reaching threshold for this line
            earliest_this_line = xdata[active_indices[-1]]  # Using -1 since time goes backwards

            earliest_activity = max(earliest_activity, earliest_this_line)

    if earliest_activity == 0:
        raise ValueError("No galaxies found reaching the mass threshold")

    # Add buffer to the range
    buffer = abs(earliest_activity) * buffer_fraction
    new_xlimit = earliest_activity + buffer

    return new_xlimit


# Example usage
if __name__ == "__main__":
    # Generate sample data with known feature importance
    np.random.seed(42)

    # Create base distribution
    n_samples = 500
    n_features = 5

    # Feature 1 and 2 are highly correlated and important
    # Feature 3 has higher variance
    # Features 4 and 5 are less important

    base_data = np.random.randn(n_samples, n_features)
    base_data[:, 1] = base_data[:, 0] + 0.5 * np.random.randn(n_samples)  # Correlated
    base_data[:, 2] = 2 * np.random.randn(n_samples)  # Higher variance

    # Create observations with outliers driven by specific features
    n_obs = 100
    observations = np.random.randn(n_obs, n_features)
    observations[:, 1] = observations[:, 0] + 0.5 * np.random.randn(n_obs)
    observations[:, 2] = 2 * np.random.randn(n_obs)

    # Add outliers driven primarily by feature 1
    outlier_indices = [10, 25, 40, 60, 80]
    observations[outlier_indices, 0] += 4  # Strong outlier in feature 1
    observations[outlier_indices, 1] += 2  # Some effect in correlated feature

    # Add outliers driven by feature 3
    observations[[15, 35, 55], 2] += 6

    feature_names = [
        "Primary_Driver",
        "Correlated_Feature",
        "High_Variance",
        "Low_Impact_1",
        "Low_Impact_2",
    ]

    # Analyze feature contributions
    print("SINGLE METHOD ANALYSIS")
    print("=" * 50)
    results = analyze_feature_contributions(
        base_data, observations, method="robust_mahalanobis", feature_names=feature_names
    )

    print("\n" + "=" * 80)

    # Compare methods
    comparison_results = compare_methods_feature_importance(
        base_data, observations, feature_names=feature_names
    )


def make_serializable(obj: Any, allowed_types=None) -> Any:
    """Recursively convert a nested dictionary/object to be JSON serializable.

    Handles common scientific computing types:
    - NumPy arrays and scalars
    - PyTorch tensors
    - JAX arrays
    - TensorFlow tensors
    - Pandas Series/DataFrames
    - Complex numbers
    - Sets
    - Bytes
    - Custom objects with __dict__

    Args:
        obj: The object to make serializable
        allowed_types: Optional list of additional types to allow (e.g., custom classes)

    Returns:
        A JSON-serializable version of the input object
    """
    # Handle None and basic JSON-serializable types

    allowed_type = [str, int, float, bool]

    if allowed_types is not None:
        allowed_type.extend(allowed_types)

    allowed_type = tuple(allowed_type)

    if obj is None or isinstance(obj, allowed_type):
        return obj

    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {str(k): make_serializable(v, allowed_types=allowed_types) for k, v in obj.items()}

    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item, allowed_types=allowed_types) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return list(obj)

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()

    # Handle complex numbers
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag, "_type": "complex"}

    # Try to import and handle NumPy types
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag), "_type": "complex"}
    except ImportError:
        pass

    # Try to handle PyTorch tensors
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except ImportError:
        pass

    # Try to handle JAX arrays
    try:
        import jax.numpy as jnp

        if hasattr(obj, "__array__") and hasattr(obj, "shape"):  # JAX array duck typing
            try:
                return jnp.asarray(obj).tolist()
            except Exception:
                pass
    except ImportError:
        pass

    # Try to handle TensorFlow tensors
    try:
        import tensorflow as tf

        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        if isinstance(obj, tf.Variable):
            return obj.numpy().tolist()
    except ImportError:
        pass

    # Try to handle Pandas objects
    try:
        import pandas as pd

        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        if isinstance(obj, pd.Index):
            return obj.tolist()
        if pd.isna(obj):  # Handle pandas NA values
            return None
    except ImportError:
        pass

    # Handle datetime objects
    try:
        from datetime import date, datetime, time

        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, time):
            return obj.isoformat()
    except ImportError:
        pass

    # Handle Decimal objects
    try:
        from decimal import Decimal

        if isinstance(obj, Decimal):
            return float(obj)
    except ImportError:
        pass

    # Handle pathlib Path objects
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except ImportError:
        pass

    # Handle UUID objects
    try:
        from uuid import UUID

        if isinstance(obj, UUID):
            return str(obj)
    except ImportError:
        pass

    # Handle custom objects with __dict__ attribute
    if hasattr(obj, "__dict__"):
        return make_serializable(obj.__dict__, allowed_types=allowed_types)

    # Handle objects with a .tolist() method (catch-all for array-like objects)
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass

    # Handle objects with .item() method (scalar array-like objects)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    # Handle iterables as a last resort (but not strings which are already handled)
    try:
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            return [make_serializable(item, allowed_types=allowed_types) for item in obj]
    except (TypeError, ValueError):
        pass

    # If all else fails, convert to string representation
    # This ensures the function doesn't crash on unknown types
    try:
        return str(obj)
    except Exception:
        return f"<unserializable object of type {type(obj).__name__}>"




def setup_mpi_named_logger(
    name: str, level: int = logging.INFO, stream: TextIO = sys.stdout
) -> logging.Logger:
    """
    Sets up a named logger that only outputs messages from MPI rank 0.

    This is more robust than configuring the root logger, as it won't
    interfere with the logging settings of other libraries.

    Args:
        name: The name for the logger instance.
        level: The logging level for the rank 0 process.
        stream: The output stream for the rank 0 process.

    Returns:
        A configured logging.Logger instance.
    """
    try:
        from mpi4py import MPI
        # Get the MPI communicator, rank, and size
        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
    except ImportError:
        # Create dummy MPI variables for single-process execution
        # This allows the script to run without mpi4py or mpiexec
        COMM = None
        RANK = 0
        SIZE = 1
    logger = logging.getLogger(name)
    
    # Prevent messages from propagating to the root logger
    logger.propagate = False

    if RANK == 0:
        # Configure the logger for the main process (rank 0)
        logger.setLevel(level)
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Add a NullHandler to silence the logger on all other processes
        logger.addHandler(logging.NullHandler())

    return logger