"""Perform error analysis on absolute error data."""
import numpy as np # For maths and arrays
from collections import namedtuple # For ErrorStatsTuple class

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
