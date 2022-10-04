"""Contains helper functions for adaptive stepsize selection."""
import numpy as np


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


def local_tolerance_scale(y_new, y_err, atol, rtol):
    """Calculates the (maximum) tolerance for each component of y (for normalising the local error)."""
    return (np.maximum(np.abs(y_new), np.abs(y_err)) * rtol) + atol


def local_error(y_new, y_err):
    """Returns the absolute error of each component of y."""
    return np.abs(y_new - y_err)


def error_norm(y_new, y_err, atol, rtol):
    """If < 1, all components of y have errors below tolerance.
    If > 1, at least one component has an error which exceeds tolerance."""
    return np.max(
        local_error(y_new, y_err) / local_tolerance_scale(y_new, y_err, atol, rtol)
    )


def calc_factor(error_norm, error_exponent, max_factor, min_factor, safety):
    """Returns the factor for calculating the new stepsize: safety * (error_norm ** error_exponent).
    Returned value will be at least min_factor and at most max_factor."""
    factor = safety * (error_norm**error_exponent)

    if error_norm < 1:
        return min(max_factor, factor)

    return max(min_factor, factor)
