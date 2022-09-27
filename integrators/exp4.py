"""A SciPy compatible implementation of the exp4 exponential integrator described in https://epubs.siam.org/doi/10.1137/S1064827595295337."""
from re import A
from scipy.sparse import csc_matrix, issparse
from scipy.optimize._numdiff import group_columns
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import (
    warn_extraneous,
    validate_max_step,
    validate_tol,
    select_initial_step,
    validate_first_step,
    num_jac,
)

import numpy as np
import inspect

from ._common import phi_step_jacob_hA, warn_max_step_exceeded


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
        jac, optional: Jacobian matrix of the right hand side of the system. Can be a constant matrix, a function which returns a matrix, or None (the default). If None then the jacobian is estimated.
        jac_sparsity, optional: If None, the Jacobian matrix is not sparse. If a matrix is passsed defining the sparsity structure of the jacobian, this is used to speed up computation (i.e. most elements of matrix will be known to be 0).
        vectorized, optional: True if fun is implemented as a vectorized function (defaults to False).
        autonomous, optional: True if fun is a function of the form fun(y) (i.e. doesn't depend on t). Defaults to False."""

    def _handle_jac(self, jac, jac_sparsity):
        """Takes the values of jac and jac_sparsity passed to __init__ and returns a function of the form jac(t, y).

        Returns:
            if jac is None, a function that estimates the Jacobian.
            if jac is constant, a dummy function which returns it.
            if jac is callable, returns it."""

        if jac is None:
            self.jac_factor = None

            if jac_sparsity is not None:
                jac_sparsity = csc_matrix(jac_sparsity)

                groups = group_columns(jac_sparsity)
                jac_sparsity = (jac_sparsity, groups)

                def estimate_wrapper(t, y):
                    self.njev += 1

                    j, self.jac_factor = num_jac(
                        self.fun_vectorized,
                        t,
                        y,
                        self.fun_single(t, y),
                        self.atol,
                        self.jac_factor,
                        jac_sparsity,
                    )

                    return j

                return estimate_wrapper

        elif callable(jac):
            # Check jacobian function accepts correct number of parameters
            if len(inspect.signature(jac).parameters) != 2:
                raise ValueError("Jacobian function must be of the form j(t, y).")

            self.njev += 1
            j = jac(self.t, self.y)

            if j.shape != (self.n, self.n):
                raise ValueError(
                    "Jacobian should have shape {0} rather than {1}.".format(
                        (self.n, self.n), j.shape
                    )
                )

            # If jacobian is sparse...
            if issparse(j):

                def call_wrapper(t, y):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=self.y.dtype)

                return call_wrapper

            else:

                def call_wrapper(t, y):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=self.y.dtype)

                return call_wrapper

        # If Jacobian is constant
        else:

            # If matrix is sparse...
            if issparse(jac):
                jac = csc_matrix(jac, dtype=self.y.dtype)

            else:
                jac = np.asarray(jac, dtype=self.y.dtype)

            if jac.shape != (self.n, self.n):
                raise ValueError(
                    "Jacobian should have shape {0} rather than {1}.".format(
                        (self.n, self.n), jac.shape
                    )
                )

            return lambda t, y: jac

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
        **extraneous
    ):
        # Raise warnings for extraneous arguments
        warn_extraneous(extraneous)

        self.autonomous = autonomous

        if autonomous:
            fun = lambda t, y: fun(y)

        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)

        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        if jac is None:
            # Method is only order 2 when jacobian is estimated
            self.order = 2
        else:
            self.order = 4

        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun,
                self.t,
                self.y,
                self.fun(self.t, self.y),
                self.direction,
                self.order,
                self.rtol,
                self.atol,
            )

        else:
            self.h_abs = validate_first_step(first_step, self.t, self.t_bound)

        self.h_abs = warn_max_step_exceeded(self.h_abs, max_step)

        self.jac = self._handle_jac(jac, jac_sparsity)

    @staticmethod
    def _calc_step(fun, A, h, t0, y0, autonomous):
        if autonomous:
            f = lambda y: fun(None, y)

        else:
            # Wrap non-autonomous function, fun(t, y), to make it an autonomous function f(y) with a t' = 1 dependency (i.e. put t into the y array)
            f = lambda y: np.array([*fun(y[-1], y[0:-1]), 1.0], dtype=y0.dtype)
            y0 = np.array([*y0, t0], dtype=y0.dtype)

        # Reused values
        hA = h * A
        phi_1_3 = phi_step_jacob_hA(hA, (1.0 / 3.0))
        phi_2_3 = phi_step_jacob_hA(hA, (2.0 / 3.0))
        phi = phi_step_jacob_hA(hA, 1.0)

        f_y0 = f(y0)

        if autonomous:
            k_1 = np.matmul(phi_1_3, f_y0)
            k_2 = np.matmul(phi_2_3, f_y0)
            k_3 = np.matmul(phi, f_y0)

        else:
            # Apply Jacobian only to original function
            k_1 = np.array([*np.matmul(phi_1_3, f_y0[0:-1]), f_y0[-1]], dtype=y0.dtype)
            k_2 = np.array([*np.matmul(phi_2_3, f_y0[0:-1]), f_y0[-1]], dtype=y0.dtype)
            k_3 = np.array([*np.matmul(phi, f_y0[0:-1]), f_y0[-1]], dtype=y0.dtype)

        w_4 = ((-7.0 / 300.0) * k_1) + ((97.0 / 150.0) * k_2) - ((37.0 / 300.0) & k_3)

        u_4 = y0 + (h * w_4)

        d_4 = f(u_4) - f_y0 - (hA * w_4)

        if autonomous:
            k_4 = np.matmul(phi_1_3, d_4)
            k_5 = np.matmul(phi_2_3, d_4)
            k_6 = np.matmul(phi, d_4)

        else:
            # Apply Jacobian only to original function
            k_4 = np.array([*np.matmul(phi_1_3, d_4[0:-1]), d_4[-1]], dtype=y0.dtype)
            k_5 = np.array([*np.matmul(phi_2_3, d_4[0:-1]), d_4[-1]], dtype=y0.dtype)
            k_6 = np.array([*np.matmul(phi, d_4[0:-1]), d_4[-1]], dtype=y0.dtype)

        w_7 = (
            ((59.0 / 300.0) * k_1)
            - ((7.0 / 75.0) * k_2)
            + ((269.0 / 300.0) * k_3) * ((2.0 / 3.0) * (k_4 + k_5 + k_6))
        )

        u_7 = y0 + h * w_7

        d_7 = f(u_7) - f_y0 - (hA * w_7)

        if autonomous:
            k_7 = np.matmul(phi_1_3, d_7)

        else:
            k_7 = np.array([*np.matmul(phi_1_3, d_7[0:-1]), d_7[-1]], dtype=y0.dtype)

        y1 = y0 + h * (k_3 + k_4 - ((4.0 / 3.0) * k_5) + k_6 + ((1.0 / 6.0) * k_7))

        if autonomous:
            return y1

        else:
            return y1[0:-1]

    def _step_impl(self):
        t_new = self.t + (self.direction * self.h_abs)

        A = self.jac(self.t, self.y)

        y_new = self._calc_step(
            self.fun, A, self.h_abs, self.t, self.y, self.autonomous
        )
