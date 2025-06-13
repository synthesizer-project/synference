
from unyt import Angstrom
import numpy as np
import scipy.stats
import sys
import re
import operator
import os
import h5py
from typing import List, Dict, Union, Callable



def load_grid_from_hdf5(
    hdf5_path: str,
    photometry_key: str = "Grid/Photometry",
    parameters_key: str = "Grid/Parameters",
    filter_codes_attr: str = "FilterCodes",
    parameters_attr: str = "ParameterNames",
    supp_key: str = "Grid/SupplementaryParameters",
    supp_attr: str = "SupplementaryParameterNames",
    supp_units_attr: str = "SupplementaryParameterUnits",
    phot_unit_attr: str = "PhotometryUnits",
) -> dict:
    """
    Load a grid from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        The loaded grid.
    """

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Load the photometry and parameters from the HDF5 file
        photometry = f[photometry_key][:]
        parameters = f[parameters_key][:]

        filter_codes = f.attrs[filter_codes_attr]
        parameter_names = f.attrs[parameters_attr]

        # Load the photometry units
        photometry_units = f.attrs[phot_unit_attr]

        output = {
            "photometry": photometry,
            "parameters": parameters,
            "filter_codes": filter_codes,
            "parameter_names": parameter_names,
            "photometry_units": photometry_units,
        }

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
    """
    Calculate the minimum and maximum wavelengths for a given redshift.
    """
    # Get the filter limits
    filter_lims = filterset.get_non_zero_lam_lims()

    # Calculate the maximum wavelength for a given redshift
    max_wav = filter_lims[1] / (1 + min_redshift)

    # Calculate the minimum wavelength for a given redshift
    min_wav = filter_lims[0] / (1 + max_redshift)

    return min_wav, max_wav


def generate_constant_R(R=300, start=1 * Angstrom, end=9e5 * Angstrom, auto_start_stop=False, filterset=None, **kwargs):
    if auto_start_stop and filterset is not None:
        start, end = calculate_min_max_wav_grid(filterset, **kwargs)

    x = [start.to(Angstrom).value]

    while x[-1] < end.to(Angstrom).value:
        x.append(x[-1] * (1.0 + 0.5 / R))

    return np.array(x) * Angstrom


def list_parameters(distribution):
    """List parameters for scipy.stats.distribution.
    # Arguments
        distribution: a string or scipy.stats distribution object.
    # Returns
        A list of distribution parameter strings.
    # from https://stackoverflow.com/questions/30453097/getting-the-parameter-names-of-scipy-stats-distributions
    """
    if isinstance(distribution, str):
        distribution = getattr(scipy.stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(",")]
    else:
        parameters = []
    if distribution.name in scipy.stats._discrete_distns._distn_names:
        parameters += ["loc"]
    elif distribution.name in scipy.stats._continuous_distns._distn_names:
        parameters += ["loc", "scale"]
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters


def rename_overlapping_parameters(lists_dict):
    """
    Check if N lists have any overlapping parameters and rename them if they do.

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
    """
    Stolen from my own code - https://github.com/tHarvey303/BD-Finder/blob/main/src/BDFit/StarFit.py

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
        self.operators = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

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

    def evaluate(self, tokens: List[str], filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Evaluate a list of tokens using provided filter data.

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
        """
        Parse and evaluate a filter arithmetic expression.

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

