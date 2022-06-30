"""Perform error analysis on absolute error data."""
import numpy as np # For maths and arrays
from collections import namedtuple # For ErrorStatsTuple class

import sys # For sys.argv
import argparse # For parsing sys.argv

def l_2_norm(x):
    """Calculates the l^2 norm of the given array (i.e. sqrt(sum(x^2))."""
    return np.sqrt(np.sum(x ** 2))

def l_infinity_norm(x):
    """Calculates the l^infinity norm of the given array (i.e. the maximum absolute value)."""
    return np.max(np.abs(x))

# Named tuple to contain the l^2 norm, l^infinity norm, and the mean of a given absolute error dataset
ErrorStatsTuple = namedtuple("ErrorStatsTuple", ["l2", "l_inf", "mean"])

def calc_err_stats(abs_err):
    """Calculates the l^2 norm, l_infinity_norm, and mean of the given array of absolute error values.
        
        Returns:
            An ErrorStatsTuple instance containing "l2", "l_inf", and "mean" in that order."""
    
    return ErrorStatsTuple(l_2_norm(abs_err), l_infinity_norm(abs_err), np.mean(abs_err))


def parse_cmd_args(args=None):
    """Parses the command line arguments and returns them, or exit the program if they are incorrect.
        
        Parameters:
            args, optional: Arguments to parse or None to use command line arguments.

        Returns:
            data_file: Path to the file which contains the absolute error data.
            graph_file: Path to save the graph to, or None if no path was specified."""

    parser = argparse.ArgumentParser(description="Performs statistical analysis on absolute error data.")

    parser.add_argument("data_file", type=str, help="File to load absolute errors from.")
    parser.add_argument("graph_file", type=str, help="File to write histogram to.", default=None, required=False)

    if args is None: # If command line arguments should be used...
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)
    
    return parsed.data_file, parsed.graph_file