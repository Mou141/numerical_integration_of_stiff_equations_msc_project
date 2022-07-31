"""Generates initial values for the Robertson Chemical Kinetics IVP that should be integratable (see robertson_ivp.py) and saves them to file."""
import numpy as np
import argparse
import sys

import robertson_ivp


def parse_args(args=None):
    """Parses command line arguments with argparse."""

    parser = argparse.ArgumentParser(
        description="Generates 'safe' initial values for the Robertson Chemical Kinets IVP and saves them to file."
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        dest="n",
        default=100,
        required=False,
        help="Number of sets of initial values to generate.",
    )

    if args is None:
        # Use sys.argv
        parsed = parser.parse_args()

    else:
        parsed = parser.parse_args(args=args)

    if parsed.n < 1:
        print("Number must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    return parsed.n


def generate_values(n):
    """Returns an array of shape (n, 3) of floating point values with randomly chosen 'safe' initial values."""

    initial_values = np.full(shape=(n, 3), fill_value=np.nan, dtype=float)

    for i in range(0, n):
        initial_values[i] = robertson_ivp.get_safe_random_y0()

    return initial_values


def main(n):
    """Saves n 'safe' sets of initial values to file 'y0.tsv'."""

    print("Generating {0} sets of initial values...".format(n))
    initial_values = generate_values(n)

    fname = "y0.tsv"

    print("Saving to file '{0}'...".format(fname))

    try:
        np.savetxt(fname, initial_values, delimiter="\t")

    except OSError as e:
        print("Error while saving file: '{0}'.".format(e), file=sys.stderr)
        sys.exit(1)

    else:
        print("File saved successfully.")


if __name__ == "__main__":
    n = parse_args()
    main(n)
