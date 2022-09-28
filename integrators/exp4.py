"""A SciPy compatible implementation of the exp4 exponential integrator described in https://epubs.siam.org/doi/10.1137/S1064827595295337."""
from scipy.sparse import csc_matrix, issparse
from scipy.optimize._numdiff import group_columns
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import (
    warn_extraneous,
    validate_max_step,
    validate_first_step,
    validate_tol,
    num_jac,
)

import numpy as np

from ._common import phi_step_jacob_hA


class EXP4(OdeSolver):
    """A subclass of the scipy.integrate.OdeSolver class which implements the exp4 numerical integration method (see https://epubs.siam.org/doi/10.1137/S1064827595295337).
    This is an exponential method of order 4 when an exact Jacobian is used, but only order 2 when the Jacobian is estimated.

    Parameters:
        fun: Function to integrate.
        t0: Initial time.
        y0: Initial value of function (i.e. y0 = fun(t0)).
        t_bound: Boundary value of t for integration. Integrator won't integrate beyond that point.
        first_step, optional: Initial stepsize to use. If not specified, an initial step will be chosen algorithmically.
        max_step, optional: Maximum allowed stepsize. Defaults to np.inf (i.e. unconstrained).
        rtol and atol, optional: Allowed relative and absolute tolerances. Default to 1e-3 and 1e-6 respectively (see scipy.integrate.solve_ivp for details).
        jac, optional: This method does not support specifying the Jacobian. Values other than None will raise a ValueError.
        jac_sparsity, optional: If None, the Jacobian matrix is not sparse. If a matrix is passsed defining the sparsity structure of the jacobian, this is used to speed up computation (i.e. most elements of matrix will be known to be 0).
        vectorized, optional: True if fun is implemented as a vectorized function (defaults to False)."""

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
        **extraneous
    ):
        # Raise warnings for extraneous arguments
        warn_extraneous(extraneous)

        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)

        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        if first_step is None:
            raise ValueError("first_step must be specified.")

        self.first_step = validate_first_step(first_step, t0, t_bound)

        if self.first_step > self.max_step:
            raise ValueError("first_step cannot exceed max_step.")

        self.jac = self.handle_jac(jac, jac_sparsity)

    def handle_jac(self, jac, sparsity):
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

        def jac_wrapper(t, y):
            self.njev += 1
            J, self.jac_factor = num_jac(
                self.wrapped_fun_vectorized,
                t,
                y,
                self.wrapped_fun_single(t, y),
                self.atol,
                self.jac_factor,
                sparsity,
            )
            return J

        return jac_wrapper

    @staticmethod
    def _calc_step(fun, A, h, y0):
        # Reused values
        hA = h * A
        phi_1_3 = phi_step_jacob_hA(hA, (1.0 / 3.0))
        phi_2_3 = phi_step_jacob_hA(hA, (2.0 / 3.0))
        phi = phi_step_jacob_hA(hA, 1.0)

        f_y0 = fun(y0)

        k_1 = np.matmul(phi_1_3, f_y0)
        k_2 = np.matmul(phi_2_3, f_y0)
        k_3 = np.matmul(phi, f_y0)

        w_4 = ((-7.0 / 300.0) * k_1) + ((97.0 / 150.0) * k_2) - ((37.0 / 300.0) * k_3)

        u_4 = y0 + (h * w_4)

        d_4 = fun(u_4) - f_y0 - np.matmul(hA, w_4)

        k_4 = np.matmul(phi_1_3, d_4)
        k_5 = np.matmul(phi_2_3, d_4)
        k_6 = np.matmul(phi, d_4)

        w_7 = (
            ((59.0 / 300.0) * k_1)
            - ((7.0 / 75.0) * k_2)
            + ((269.0 / 300.0) * k_3) * ((2.0 / 3.0) * (k_4 + k_5 + k_6))
        )

        u_7 = y0 + h * w_7

        d_7 = fun(u_7) - f_y0 - np.matmul(hA, w_7)

        k_7 = np.matmul(phi_1_3, d_7)

        y1 = y0 + h * (k_3 + k_4 - ((4.0 / 3.0) * k_5) + k_6 + ((1.0 / 6.0) * k_7))

        return y1

    @staticmethod
    def add_dependency(fun, t, y):
        """Takes the return value of the specified function and adds an explicit t' = 1 dependency to it.

        NB: Must use the non-vectorized version of the function, i.e. self.fun (if self.nfev is to be incremented) or self.fun_single (if not)."""

        d = fun(t, y)
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
        # Shrink stepsize if it goes beyond edge of integration bounds
        self.h_abs = np.min(self.first_step, np.abs(self.t_bound - self.t))

        # Add the t dependency to y
        y_wrapped = np.array([*self.y, self.t], dtype=self.y.dtype)

        # Estimate the Jacobian matrix
        A = self.jac(self.t, y_wrapped)

        t_new = self.t + (self.direction * self.h_abs)
        y_wrapped_new = self._calc_step(self.wrapped_fun, A, self.h_abs, y_wrapped)

        self.t = t_new

        # Unwrap new y (i.e. cut t off end of array)
        self.y = y_wrapped_new[0:-1]

        return True, None
