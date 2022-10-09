"""A SciPy compatible implementation of the exp4 exponential integrator described in https://epubs.siam.org/doi/10.1137/S1064827595295337."""
from scipy.sparse import csc_matrix, issparse
from scipy.optimize._numdiff import group_columns
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import (
    warn_extraneous,
    validate_max_step,
    validate_first_step,
    select_initial_step,
    validate_tol,
    num_jac,
)

import numpy as np

from ._common import phi_step_jacob_hA
from ._stepsize import calc_min_step, error_norm, calc_factor, stepsize_check


class EXP4(OdeSolver):
    """A subclass of the scipy.integrate.OdeSolver class which implements the exp4 numerical integration method (see https://epubs.siam.org/doi/10.1137/S1064827595295337).
    This is an exponential method of order 4 when an exact Jacobian is used, but only order 2 when the Jacobian is estimated.

    Parameters:
        fun: Function to integrate of form fun(t, y).
        t0: Initial time.
        y0: Initial value of function (i.e. y0 = fun(t0)).
        t_bound: Boundary value of t for integration. Integrator won't integrate beyond that point.
        first_step, optional: Initial stepsize to use. If not specified, an initial step will be chosen algorithmically.
        max_step, optional: Maximum allowed stepsize. Defaults to np.inf (i.e. unconstrained).
        rtol and atol, optional: Allowed relative and absolute tolerances. Default to 1e-3 and 1e-6 respectively (see scipy.integrate.solve_ivp for details).
        jac, optional: This method does not support specifying the Jacobian. Values other than None will raise a ValueError.
        jac_sparsity, optional: If None, the Jacobian matrix is not sparse. If a matrix is passsed defining the sparsity structure of the jacobian, this is used to speed up computation (i.e. most elements of matrix will be known to be 0).
        vectorized, optional: True if fun is implemented as a vectorized function (defaults to False).
        autonomous, optional: True if fun(t, y) depends only on y (defaults to False). Setting to True if fun(t, y) is autonomous will increase execution speed but not accuracy.
        max_factor, optional: Maximum amount by which stepsize may increase after a successful step (defaults to 10.0).
        min_factor, optional: Minimum amount stepsize can shrink by after an unsuccessful step (defaults to 0.2).
        safety, optional: Scaling factor for calculating new stepsizes. Factor to shrink calculated stepsizes to increase convergence rate. Should be < 1 (defaults to 0.9)."""

    # Order of the embedded method used to estimate the error
    error_estimation_order = 1.0

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        first_step=None,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
        jac=None,
        jac_sparsity=None,
        vectorized=False,
        autonomous=False,
        max_factor=10.0,
        min_factor=0.2,
        safety=0.9,
        **extraneous
    ):
        # Raise warnings for extraneous arguments
        warn_extraneous(extraneous)

        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)

        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        self.autonomous = autonomous

        if max_factor <= 1.0:
            raise ValueError("max_factor must be greater than 1.")

        if min_factor <= 0.0:
            raise ValueError("min_factor must be greater than 0.0.")

        if safety > 1.0 or safety <= 0.0:
            raise ValueError(
                "safety must be greater than 0 and less than or equal to 1."
            )

        self.max_factor = max_factor
        self.min_factor = min_factor
        self.safety = safety

        if first_step is None:
            # Need to select stepsize
            self.h = self.direction * select_initial_step(
                fun,
                t0,
                y0,
                fun(t0, y0),
                self.direction,
                self.error_estimation_order,
                self.rtol,
                self.atol,
            )

        else:
            self.h = self.direction * validate_first_step(first_step, t0, t_bound)

        if np.abs(self.h) > self.max_step:
            raise ValueError(
                "First step {0} exceeds maximum step {1}.".format(
                    first_step, self.max_step
                )
            )

        if np.abs(self.h) < calc_min_step(t0, t_bound, self.direction):
            raise ValueError("First step is too small.")

        self.h_old = None
        self.err_old = None

        self.jac = self.handle_jac(jac, jac_sparsity, autonomous)

    def handle_jac(self, jac, sparsity, autonomous):
        """Wraps the num_jac function that estimates the Jacobian at each step."""

        if jac is not None:
            raise ValueError(
                "This method does not currently support specifying an explicit Jacobian."
            )

        if sparsity is not None:
            if issparse(sparsity):
                sparsity = csc_matrix(sparsity)

            groups = group_columns(sparsity)
            sparsity = (sparsity, groups)

        self.jac_factor = None

        if autonomous:
            vec_fun = self.fun_vectorized
            single_fun = self.fun_single

        else:
            # Add t' = 1 dependency
            vec_fun = self.wrapped_fun_vectorized
            single_fun = self.wrapped_fun_single

        def jac_wrapper(t, y):
            self.njev += 1
            J, self.jac_factor = num_jac(
                vec_fun,
                t,
                y,
                single_fun(t, y),
                self.atol,
                self.jac_factor,
                sparsity,
            )
            return J

        return jac_wrapper

    @staticmethod
    def _embedded_error_step(y0, h, k_3, k_4, k_5, k_6, k_7):
        """Performs a step of the embedded method used for error calculation."""
        return y0 + h * (
            k_3 + (-0.5 * k_4) + ((-2.0 / 3.0) + k_5) + (0.5 * k_6) + (0.5 * k_7)
        )

    @classmethod
    def _calc_step(cls, fun, A, h, y0):
        """Performs a step of the exp4 method and the embedded error calculation method."""
        # Reused values
        hA = h * A
        phi_1_3 = phi_step_jacob_hA(hA, (1.0 / 3.0))
        phi_2_3 = phi_step_jacob_hA(hA, (2.0 / 3.0))
        phi = phi_step_jacob_hA(hA, 1.0)

        f_y0 = fun(None, y0)

        k_1 = np.matmul(phi_1_3, f_y0)
        k_2 = np.matmul(phi_2_3, f_y0)
        k_3 = np.matmul(phi, f_y0)

        w_4 = ((-7.0 / 300.0) * k_1) + ((97.0 / 150.0) * k_2) - ((37.0 / 300.0) * k_3)

        u_4 = y0 + (h * w_4)

        d_4 = fun(None, u_4) - f_y0 - np.matmul(hA, w_4)

        k_4 = np.matmul(phi_1_3, d_4)
        k_5 = np.matmul(phi_2_3, d_4)
        k_6 = np.matmul(phi, d_4)

        w_7 = (
            ((59.0 / 300.0) * k_1)
            - ((7.0 / 75.0) * k_2)
            + ((269.0 / 300.0) * k_3) * ((2.0 / 3.0) * (k_4 + k_5 + k_6))
        )

        u_7 = y0 + h * w_7

        d_7 = fun(None, u_7) - f_y0 - np.matmul(hA, w_7)

        k_7 = np.matmul(phi_1_3, d_7)

        y1 = y0 + h * (k_3 + k_4 - ((4.0 / 3.0) * k_5) + k_6 + ((1.0 / 6.0) * k_7))

        # Perform embedded error step
        y_err = cls._embedded_error_step(y0, h, k_3, k_4, k_5, k_6, k_7)

        return y1, y_err

    @staticmethod
    def add_dependency(fun, t, y):
        """Takes the return value of the specified function and adds an explicit t' = 1 dependency to it.

        NB: Must use the non-vectorized version of the function, i.e. self.fun (if self.nfev is to be incremented) or self.fun_single (if not)."""

        # Extract y and t from the wrapped y array, execute the function,
        # and make sure the result is an array of at least 1D
        d = np.array(fun(y[-1], y[0:-1]), ndmin=1, copy=False)

        return np.array([*d, 1.0], dtype=d.dtype)

    def wrapped_fun(self, t, y):
        """Version of self.fun with explicit t' = 1 dependency."""

        return self.add_dependency(self.fun, t, y)

    def wrapped_fun_single(self, t, y):
        """Version of self.fun_single with explicit t' = 1 dependency."""

        return self.add_dependency(self.fun_single, t, y)

    def wrapped_fun_vectorized(self, t, y):
        """Version of self.fun_vectorized with explicit t' = 1 dependency."""
        # NB: Vectorization algorithm copied from scipy.integrate._ivp.base.py (i.e same one base class uses)

        d = np.empty_like(y)

        for i, yi in enumerate(y.T):
            d[:, i] = self.wrapped_fun_single(t, yi)

        return d

    def _step_impl(self):
        if self.autonomous:
            fun = self.fun
            y_old = self.y

        else:
            # Wrap function and y to add t dependency
            fun = self.wrapped_fun
            y_old = np.array([*self.y, self.t], dtype=self.y.dtype)

        # Estimate Jacobian matrix at current values of t and y
        A = self.jac(self.t, y_old)

        while True:
            # Make sure stepsize won't take integrator beyond end of bounds
            self.h = stepsize_check(self.h, self.t, self.t_bound, self.direction)

            t_new = self.t + self.h

            # Perform one step of exp4 and one step of embedded error method
            y_new, y_err = self._calc_step(fun, A, self.h, y_old)

            # If all values of y_new are within tolerance, err < 1
            # If any values are above tolerance, err > 1
            if self.autonomous:
                err = error_norm(y_new, y_err, y_old, self.atol, self.rtol)
            else:
                err = error_norm(
                    y_new[0:-1], y_err[0:-1], y_old[0:-1], self.atol, self.rtol
                )

            # Factor to alter steppsize by (shrink if step not accurate, grow otherwise)
            factor = calc_factor(
                self.h,
                self.h_old,
                err,
                self.err_old,
                self.min_factor,
                self.max_factor,
                self.safety,
            )

            self.h_old = self.h
            self.err_old = self.err

            self.h = factor * self.h

            # All errors are within tolerance, solution has converged at this step
            if err < 1.0:
                self.t = t_new

                if self.autonomous:
                    self.y = y_new
                else:
                    self.y = y_new[0:-1]

                return True, None
