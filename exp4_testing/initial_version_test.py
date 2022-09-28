"""Python script to test the initial impementation of integrators.exp4.EXP4 (constant time-step)
by integrating a non-stiff differential system."""

from pathlib import Path
import sys

# Add path of parent folder to sys.path so that exp4 can be imported
PARENT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT_PATH))

from integrators import exp4

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def test_function(t, y):
    """Test function to integrate (2e^2x)."""
    return 2.0 * np.exp(2.0 * t)


def test_solution(t):
    """Analytical solution of test function (e^2x)."""
    return np.exp(2.0 * t)


# Initial value of above differential system
TEST_Y0 = 2.0
TEST_T0 = 0.0


def main(start_t, end_t, stepsize, y0, fun, sol):
    print("Parameters:")
    print("\tt_start: {0}".format(start_t))
    print("\tt_end: {0}".format(end_t))
    print("\tStep Size: {0}".format(stepsize))
    print("\ty0: {0}".format(y0))

    print("Performing integration...")
    results = solve_ivp(
        fun, (start_t, end_t), y0, method=exp4.EXP4, first_step=stepsize
    )

    if results.success:
        print("Integration succeeded.")

    else:
        print("Integration failed.")
        sys.exit(1)

    print("\tnfev: {0}".format(results.nfev))
    print("\tnjev: {0}".format(results.njev))

    # Exact solution
    t_exact = np.linspace(start_t, end_t, 10000)
    y_exact = sol(t_exact)

    plt.plot(results.t, results.y, label="Numerical Solution")
    plt.plot(t_exact, y_exact, label="Analytical Solution")
    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main(TEST_T0, 10.0, 0.1, np.array([TEST_Y0]), test_function, test_solution)
