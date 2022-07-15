"""Contains the Robertson Chemical Kinetics Stiff Initial Value Problem."""
import stiff_functions
import numpy as np


def initial_value_problem(t, y, k1=0.04, k2=1.0e04, k3=3.0e07):
    """The Robertson Chemical Kinetics stiff IVP."""

    # Unpack dimensions of IVP for readability
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    dy1 = (k1 * y1) + (k2 * y2 * y3)
    dy2 = -(k1 * y1) - (k2 * y2 * y3) - (k3 * (y2**2))
    dy3 = k3 * (y2**2)

    return np.array([dy1, dy2, dy3])


# Test version of the Robertston IVP that has fixed values for y0 (for repeatability)
TEST_IVP = stiff_functions.IVPTuple(
    initial_value_problem, None, y0=np.array([0.25, 0.25, 0.5]), t0=0.0
)


def get_random_y0():
    """Returns a y0 array (y1(0), y2(0) and y3(0)) for the Robertson IVP that contains 3 random numbers such that sum(y0) == 1.0."""
    rng = np.random.default_rng()

    # Two random numbers uniformly chosen between 0.0 and 0.5 and the third is what's left to make sum(y0) == 1.0
    y01 = rng.uniform(0.0, 0.5)
    y02 = rng.uniform(0.0, 0.5)
    y03 = 1.0 - (y01 + y02)

    y0 = np.array([y01, y02, y03])

    rng.shuffle(y0)  # Assign random numbers to y0 indices at random

    return y0


def get_robertson_ivp():
    """Returns an instance of stiff_functions.IVPTuple for the Robertson Chemical Kinetics IVP with randomly chosen y0 such that sum(y0) == 1.0."""

    y0 = get_random_y0()

    return stiff_functions.IVPTuple(initial_value_problem, None, y0=y0, t0=0.0)
