"""Contains the Robertson Chemical Kinetics Stiff Initial Value Problem."""
import stiff_functions
import numpy as np

def initial_value_problem(t, y, k1=0.04, k2=1.0E-04, k3=3.0E7):
    """The Robertson Chemical Kinetics stiff IVP."""

    # Unpack dimensions of IVP for readability
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    dy1 = (k1 * y1) + (k2 * y2 * y3)
    dy2 = -(k1 * y1) - (k2 * y2 * y3) - (k3 *(y2 ** 2))
    dy3 = k3 * (y2 ** 2)

    return np.array([y1, y2, y3])

# Test version of the Robertston IVP that has fixed values for y0 (for repeatability)
TEST_IVP = stiff_functions.IVPTuple(initial_value_problem, None, y0=np.array([0.25, 0.25, 0.5]), t0=0.0)