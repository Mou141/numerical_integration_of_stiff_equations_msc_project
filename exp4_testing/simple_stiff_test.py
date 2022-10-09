"""Tests the exp4 implementation with the simple 1D stiff IVP in stiff_functions.STIFF_IVP."""

from pathlib import Path
import sys

# Path to parent folder of folder this file is in
PARENT_PATH = Path(__file__).resolve().parent.parent

# Path to prelim folder
PRELIM_PATH = PARENT_PATH / "prelim"

sys.path.extend([str(PARENT_PATH), str(PRELIM_PATH)])

from integrators import exp4
from stiff_functions import STIFF_IVP

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def main():
    t_span = (STIFF_IVP.t0, 10.0)

    print("Integrating...")
    results = solve_ivp(
        STIFF_IVP.ODEFunction, t_span, STIFF_IVP.y0, method=exp4.EXP4, autonomous=True
    )

    if not results.success:
        print("Integration Failed: '{0}'.".format(results.message))
        sys.exit(1)

    print("Integration Successful.")
    print("\tnjev: {0}".format(results.njev))
    print("\tnfev: {0}".format(results.nfev))
    print("\tnpoints: {0}".format(len(results.t)))

    y_exact = STIFF_IVP.SolutionFunction(results.t)

    plt.plot(results.t, results.y[0], "bo", label="Numerical Solution")
    plt.plot(results.t, y_exact, label="Analytical Solution")

    plt.show()


if __name__ == "__main__":
    main()
