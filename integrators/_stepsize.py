"""Contains helper functions for adaptive stepsize selection."""
import numpy as np

# RMS norm
from scipy.integrate._ivp.common import norm


def calc_min_step(t, t_bound, direction, factor=10.0):
    """Determines the smallest possible step that can be taken at this point.

    Parameters:
        t: Time at current position.
        t_bound: Boundary of integration range.
        direction: Direction of integration.
        factor, optional: Safety factor that the minimum possible stepsize is multiplied by (defaults to 10).

    Returns:
        factor * (minimum possible step from t) or abs(t_bound - t). Whichever is smaller."""

    # factor * the next smallest value that moves from t in the given direction (+ or -)
    min_step = factor * np.abs(np.nextafter(t, direction * np.inf) - t)

    # Step that would take us to end of integration range
    bound_step = np.abs(t_bound - t)

    return min(min_step, bound_step)


def local_tolerance_scale(y_new, y_old, atol, rtol):
    """Calculates the (maximum) tolerance for each component of y (for normalising the local error)."""
    return atol + (np.maximum(np.abs(y_new), np.abs(y_old)) * rtol)


def local_error(y_new, y_err):
    """Returns the linear error of each component of y."""
    return y_new - y_err


def error_norm(y_new, y_err, y_old, atol, rtol):
    """RMS norm of error."""
    return norm(
        local_error(y_new, y_err) / local_tolerance_scale(y_new, y_old, atol, rtol)
    )


def stepsize_check(h, t, t_bound, direction):
    """If current stepsize would take integrator beyond end of bounds,
    shrink it so that it will reach end of bounds."""
    # Calculate step that would take us to end of solution
    abs_bound = np.abs(t_bound - t)

    if np.abs(h) > abs_bound:
        return direction * abs_bound
    else:
        return h


def calc_factor(h, h_old, err, err_old, min_factor, max_factor, safety):
    """Calculates the factor to change the stepsize by according to the method used by Radau5."""

    if err == 0.0 or err_old is None or h_old is None:
        multiplier = 1.0

    else:
        multiplier = np.abs(h / h_old) * ((err_old / err) ** 0.25)

    multiplier = min(1.0, multiplier)

    # Can ignore divide by 0 because np.inf will be handled by min()
    with np.errstate(divide="ignore"):
        factor = multiplier * (err**-0.25)

    factor = max(min_factor, (factor * safety))

    factor = min(max_factor, factor)

    return factor
