"""Tries to integrate the Robertson IVP with random sets of initial values until a set that works with the exp4 implementation is found."""

import sys

from robertson_test import PARENT_PATH, PRELIM_PATH, HOCHBRUCH_T_SPAN

# Add parent folder to this folder and prelim folder to sys.path
sys.path.extend([str(PARENT_PATH), str(PRELIM_PATH)])

from integrators import exp4
from robertson_ivp import initial_value_problem, get_random_y0

from scipy.integrate import solve_ivp


def test_initial_values(y0):
    results = solve_ivp(
        initial_value_problem, HOCHBRUCH_T_SPAN, y0, method=exp4.EXP4, autonomous=True
    )
    return results.success


def main():
    success = False
    y0 = None

    while not success:
        y0 = get_random_y0()

        print("Trying: {0}...".format(y0))

        success = test_initial_values(y0)

    print("\n\nSuccessful initial values found: {0}".format(y0))


if __name__ == "__main__":
    main()
