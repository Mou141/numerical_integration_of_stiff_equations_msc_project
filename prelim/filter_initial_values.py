"""Takes a tsv file containing sets of initial values and another containing l2 or l_inf as produced by robertson_bulk_error.py, and filters out the initial values that cannot be integrated."""

import numpy as np
import argparse
import sys


def parse_args(args=None):
    """Parses the command line arguments specified."""

    parser = argparse.ArgumentParser(
        description="Filters out y0 values from a file that could not be integrated."
    )

    parser.add_argument(
        "Y0_FILE",
        type=str,
        required=True,
        dest="y0_file",
        help="File containing the y0 values to filter.",
    )
    parser.add_argument(
        "FILTER_FILE",
        type=str,
        required=True,
        dest="filter_file",
        help="File containing the l2 or l_inf values to filter the y0 values with.",
    )

    if args is None:
        # use sys.argv
        parsed = parser.parse_args()

    else:
        parsed = parser.parse_args(args=args)

    return parsed.y0_file, parsed.filter_file


def load_data(y0_file, filter_file):
    """Loads y0 data from y0_file and nan_array from filter_file and then checks that the arrays are the correct shapes and lengths."""

    y0 = np.loadtxt(y0_file)
    nan_array = np.loadtxt(filter_file)

    if len(y0.shape) != 2:
        raise ValueError(
            "Data in '{0}' is a {1} dimensional array (should have 2 dimensions).".format(
                y0_file, len(y0.shape)
            )
        )

    if len(nan_array.shape) != 1:
        raise ValueError(
            "Data in '{0}' is a {1} dimensional array (should have 1 dimension).".format(
                filter_file, len(nan_array.shape)
            )
        )

    if y0.shape[0] != len(nan_array):
        raise ValueError(
            "Files '{0}' and '{1}' should be the same length (rather than {2} and {3}).".format(
                y0_file, filter_file, y0.shape[0], len(nan_array)
            )
        )

    return y0, nan_array


def filter_y0(y0, nan_array):
    """Filters out the indices of y0 where nan_array contains nan_values.

    Arguments:
        y0: Array of shape (n, 3)
        nan_array: Array of shape (n,)

    Returns:
        Array of shape (k, 3)"""

    # True where nan_array does not contain nan values
    mask = np.logical_not(np.isnan(nan_array))

    return y0[:][mask]


def main(y0_file, filter_file, dest_file="y0.tsv"):
    """Takes the initial values in y0_file, filters them with the data in filter_file, and then saves the filtered values to dest_file (defaults to 'y0.tsv')."""

    try:
        print("Loading data from '{0}' and '{1}'...".format(y0_file, filter_file))
        y0, nan_array = load_data(y0_file, filter_file)

    except (ValueError, IOError, OSError) as e:
        print("Error while loading data: '{0}'.".format(e), file=sys.stderr)
        sys.exit(1)

    else:
        print("Data loaded.")

    filtered_y0 = filter_y0(y0, nan_array)

    try:
        print("Saving filtered y0 values to '{0}'...".format(dest_file))
        np.savetxt(dest_file, filtered_y0, delimiter="\t")

    except (IOError, OSError) as e:
        print("Error while saving data: '{0}'.".format(e), file=sys.stderr)
        sys.exit(1)

    else:
        print("Data saved successfully.")


if __name__ == "__main__":
    y0_file, filter_file = parse_args()
    main(y0_file, filter_file)
