"""Tests the exp4 implementation on the Robertson Chemical Kinetics IVP."""

from pathlib import Path
import sys

# Path of parent folder to the one this file is in
PARENT_PATH = Path(__file__).resolve().parent.parent

# Path of "prelim" folder
PRELIM_PATH = PARENT_PATH / "prelim"

# Add both paths to sys.path so integrators.exp4 and robertson_ivp can be imported
sys.path.extend([str(PARENT_PATH), str(PRELIM_PATH)])

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from integrators import exp4

# Import robertson problem with static initial values for testing
from robertson_ivp import TEST_IVP

# Sum of y values in solution should be equal to 1.0 (get_constraint calculates this)
from robertson_integration import get_constraint, save_data

from calc_error import linear_error

HAIRER_Y0 = np.array([1.0, 0.0, 0.0])
HOCHBRUCH_T_SPAN = (0.0, 400)


def integate_robertson(
    y0=TEST_IVP.y0,
    t_bounds=(TEST_IVP.t0, 1.0e4),
    max_factor=10.0,
    min_factor=0.2,
    safety=0.9,
    atol=1e-6,
    rtol=1e-3,
):
    """Integrates the Robertson Chemical Kinetics IVP with the integrators.exp4.EXP4 integrator."""

    print("Parameters:")
    print("\ty0: {0}".format(y0))
    print("\tt: {0} to {1}".format(*t_bounds))
    print("\tmax_factor: {0}".format(max_factor))
    print("\tmin_factor: {0}".format(min_factor))
    print("\tsafety: {0}".format(safety))
    print("\tatol: {0}".format(atol))
    print("\trtol: {0}".format(rtol))

    print("Integrating...")

    results = solve_ivp(
        TEST_IVP.ODEFunction,
        t_bounds,
        y0,
        method=exp4.EXP4,
        max_factor=max_factor,
        min_factor=min_factor,
        safety=safety,
        atol=atol,
        rtol=rtol,
        autonomous=True,
    )

    if not results.success:
        print("Integration failed: '{0}'.".format(results.message), file=sys.stderr)
        sys.exit(1)

    print("Integration Complete.")
    print("\tnjev: {0}".format(results.njev))
    print("\tnfev: {0}".format(results.nfev))
    print("\tnpoints: {0}".format(len(results.t)))

    return results.t, results.y


def main():
    t, y = integate_robertson(y0=HAIRER_Y0, t_bounds=HOCHBRUCH_T_SPAN)

    constraint = get_constraint(y)
    lin_err = linear_error(constraint, 1.0)

    save_data("exp4", t, y, lin_err)

    plt.plot(t, lin_err, label="Linear Error")
    plt.show()


if __name__ == "__main__":
    main()
