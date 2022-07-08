# Test suite for ../calc_error.py

from pathlib import Path
import sys # For sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add the parent directory of the directory that this file to the python path (since this contains calc_error.py)

import contextlib
import os

import pytest

import calc_error
import stiff_functions

import numpy as np

class TestErrors:
    """Tests for absolute and relative errror calculations."""
    
    # Test values for test_linear_error
    lin_test_1 = [1.0, 0.5, 0.5] # Scalar test with positive answer
    lin_test_2 = [1.0, 1.5, -0.5] # Scalar test with negative answer
    lin_test_3 = [-1.0, -0.5, -0.5] # Scalar test with negative arguments
    lin_test_4 = [np.array([1.0, 1.0, -1.0]), np.array([0.5, 1.5, -0.5]), np.array([0.5, -0.5, -0.5])] # 1D array test combining above
    lin_test_5 = [np.array([[1.0, 1.0, -1.0], [2.4, 5.6, -7.6]]), np.array([[0.5, 1.5, -0.5], [4.5, -7.6, -54.5]]), np.array([[0.5, -0.5, -0.5], [-2.1, 13.2, 46.9]])] # 2D array test

    @pytest.mark.parametrize("y_num,y_exact,lin_err", [lin_test_1, lin_test_2, lin_test_3, lin_test_4, lin_test_5])
    def test_linear_error(self, y_num, y_exact, lin_err):
        """Tests calc_error.linear_error by calculating linear errors and comparing to precalculated values."""
        assert calc_error.linear_error(y_num, y_exact) == pytest.approx(lin_err)
    
    # Test values for test_fractional_error
    frac_test_1 = [3.0, 1.0, (1.0/3.0)] # Scalar test with positive arguments
    frac_test_2 = [-3.0, 1.0, (1.0/3.0)] # Scalar test with negative y_num
    frac_test_3 = [3.0, -1.0, (1.0/3.0)] # Scalar test with negative lin_err
    frac_test_4 = [3.0, np.nan, np.nan] # Scalar test with NaN
    frac_test_5 = [np.array([3.0, -3.0]), np.array([1.0, 1.0]), np.array([(1.0/3.0), (1.0/3.0)])] # 1D array test
    frac_test_6 = [np.array([[3.0, -3.0], [6.7, -8.4]]), np.array([[1.0, 1.0], [5.6, 6.6]]), np.array([[(1.0/3.0), (1.0/3.0)], [(5.6/6.7), (6.6/8.4)]])] # 2D array test

    @pytest.mark("y_num,lin_err,f_err", [frac_test_1, frac_test_2, frac_test_3, frac_test_4, frac_test_5, frac_test_6])
    def test_fractional_error(self, y_num, lin_err, f_err):
        """Tests calc_error.fractional_error by asserting that calc_error.fractional_error(lin_err, y_num) == f_err."""
        assert calc_error.fractional_error(lin_err, y_num) == pytest.approx(f_err, nan_ok=True) # pytest.approx copes with floating point arithmetic issues

class TestFiles:
    """Performs tests on save_error function and resulting files.
    """
    
    delete_after = True # True if files should be deleted after tests complete. Else, False
    
    @staticmethod
    def _cleanup(paths):
        """Cleanup method to delete generated files after tests complete."""
        for p in paths: # For each path...
            with contextlib.suppress(FileNotFoundError, OSError):
                os.remove(p) # Delete the file, ignoring errors if the file does not exist or an OSError occurs (but not if p is a directory)
    
    @pytest.mark.parametrize("method,t,lin_err,frac_err,y", [("1DTest", np.array([1.0, 2.0, 3.0]), np.array([565.32, 13.622, 2322.2]), np.array([12313.22, 232.22, 12312.22]), np.array([645646.0, 3433.1, -45.2])), ("2DTest", np.array([12.1, -5.6, 6.7]), np.array([[56.75756767, -454.43434, -5.6], [7.853, 3535.42, -676.6]]), np.array([[56.7, 454.4, -5.6], [87877.853, -353589.42, 6769.6]]), np.array([[5697.07, 454.487, -5878.6], [-7887.853, 35676.42, 976.6]]))]) # Check save_error for 1D and 2D y
    def test_save_error(self, method, t, lin_err, frac_err, y):
        """Tests the calc_error.save_error function.
               
            Tests:
                - Files save without raising exception.
                - Files exist afterwards.
                - Files can be loaded back into program by numpy.
                - Arrays from files have same length and shape as originals.
                - Arrays from files contain same values as originals."""
        
        paths = [] # Set paths to an empty list so that cleanup code will not throw error if no files are generated
        
        try:
            paths = calc_error.save_error(method, t, lin_err, frac_err, y) # Save the arrays to file (test will fail if any exception thrown)

            for p, arr in zip(paths, [t, lin_err.T, frac_err.T, y.T]): # For each path, get the path and the associated array (transposing the errors and y so they match the file contents)...
                assert Path(p).is_file() # Make sure that the path points to a file that exists

                saved_array = np.loadtxt(p) # Load the array from file
                assert len(saved_array) == len(arr) # Check that it has the same length as the original array
                assert np.shape(saved_array) == np.shape(arr) # Check that it has the same shape as the original array
                assert saved_array == pytest.approx(arr, nan_ok=True) # Check arrays are equal to each other (use approx incase savetxt clips decimal places)
        finally:
            if self.delete_after: # If files should be deleted after completion of tests...
                self._cleanup(paths)

class TestIntegration:
    """Tests the main body of the program by integrating both the 1D and 2D stiff IVPs, calculating the errors, and plotting the fractional error on a graph.
    However, output to file is disabled."""
    
    # Integrate 1D IVP between 0.0 and 1.0 and the 2D IVP between 0.0 and 100.0 for each of the SciPy stiff methods
    @pytest.mark.parametrize("ivp,end_t", [(stiff_functions.STIFF_IVP, 1.0), (stiff_functions.STIFF_IVP2, 100.0)])
    def test_test_integrator_errors(self, ivp, end_t):
        """Call calc_error.test_integrator_errors with standard list of methods, given initial value problem, given final value of t, and with output set to False."""
        calc_error.test_integrator_errors(end_t=end_t, ivp=ivp)