"""Counts the number of evaluations of the Jacobian matrix necessary to achieve given relative tolerances."""

from measure_robertson_precision import TEST_IVP, T_SPAN, METHODS, RTOL_RANGE

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def measure_all_methods(methods, t_span, rtol_range):
    out = {}

    for method_name, method_class in methods:
        njevs = np.empty(shape=rtol_range.shape, dtype=int)

        for i, rtol in enumerate(rtol_range):
            print("Integrating {0} (rtol : {1})...".format(method_name, rtol))
            results = solve_ivp(
                TEST_IVP.ODEFunction,
                t_span,
                TEST_IVP.y0,
                method=method_class,
                rtol=rtol,
            )

            if results.success:
                print("Integration succeeded.")
                njevs[i] = results.njev
            else:
                print("Integration failed.")
                njevs[i] = np.inf

            out[method_name] = njevs

    return out


def main():
    results = measure_all_methods(METHODS, T_SPAN, RTOL_RANGE)

    plt.xlabel("Relative Tolerance")
    plt.ylabel("Number of Jacobian Evaluations")

    plt.xscale("log")

    for method_name, njevs in results.items():
        plt.plot(RTOL_RANGE, njevs, label=method_name, marker="o")

    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
