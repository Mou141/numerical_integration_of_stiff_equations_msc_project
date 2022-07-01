"""Perform error analysis on absolute error data."""
import numpy as np # For maths and arrays
from collections import namedtuple # For ErrorStatsTuple class

import argparse # For parsing sys.argv

import matplotlib.pyplot as plt # For plotting histograms

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
    parser.add_argument("--graph-file", dest="graph_file", type=str, help="File to write histogram to.", default=None, required=False)

    if args is None: # If command line arguments should be used...
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)
    
    return parsed.data_file, parsed.graph_file

def read_data_file(file_path):
    """Reads absolute error data from the specified file and returns a 2D array organised by dimension."""
    
    data = np.loadtxt(file_path) # Read the data from file

    if len(data.shape) == 1: # If the data contains only one dimension, then data is from a 1D IVP...
        return np.array([data]) # Wrap it in another array so that it is a 2D array

    else: # If data is 2D...
        return data.T # Transpose it so that columns and rows are switched (i.e. data[0] will return the errors in y1, data[1] the errors in y[2] etc.)

def make_histogram(data, file_path=None, bins=10):
    """Plots histograms of the absolute error data, one histogram for each dimension.
        
        Arguments:
            data: The absolute error data arranged by dimension.
            file_path, optional: The file to save the graphs to, or None if the graph should be displayed instead (default: None).
            bins, optional: The number of bins to place the histogram values in (default: 10)."""

    ndim = len(data[:]) # Get the number of dimensions in the error data

    figure, ax = plt.subplots(0, ndim, sharey=True) # Create a list of subplots, one for each dimension of the error data. All the subplots are on the same row and share a y-axis

    if ndim == 1: # If there is only one dimension...
        ax.set_xlabel("err(y)") # Set the label for the x-axis
        ax = [ax] # Put the axis in a list since only one will be returned by the plt.subplots function

    else: # If there is more than one dimension...
        for d in range(0, ndim): # For each dimension...
            ax[d].set_xlabel("err(y{0})".format(d+1)) # Set the label for the x-axis

    ax[0].set_ylabel("N") # Set the label for the y-axis (only one since the axis is shared)

    # For each dimension, plot the errors in that dimension on a histogram
    for d in range(0, ndim):
        ax[d].hist(data[d], bins=bins)
    
    if file_path is None: # If no file is specified...
        plt.show() # Display the histograms

    else: # If a file is specified...
        plt.savefig(file_path)
