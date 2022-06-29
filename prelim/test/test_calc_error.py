# Test suite for ../calc_error.py

from pathlib import Path
import sys # For sys.path
sys.path.append(Path(__file__).resolve().parent.parent) # Add the parent directory of the directory that this file to the python path (since this contains calc_error.py)

import contextlib
import os

import pytest
import calc_error

import numpy as np

class ErrorTests:
    """Tests for absolute and relative errror calculations."""
    
    @pytest.mark.parametrize("y_num,y_exact,a_err", [(1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (np.array([1.0, 1.0, 0.5]), np.array([1.0, -1.0, 2.5]), np.array([0.0, 0.0, 2.0]))]) # Test values and outputs to use (test one value that would be the same with and without abs, one that needs abs, and one array to check it works properly with arrays)
    def test_absolute_error(self, y_num, y_exact, a_err):
        """Tests calc_error.absolute_error by asserting that calc_error.absolute_error(y_num, y_exact) == a_err."""
        assert calc_error.absolute_error(y_num, y_exact) == pytest.approx(a_err) # pytest.approx copes with floating point arithmetic issues
    
    @pytest.mark.parametrize("y_num,a_err,f_err", [(5.0, 2.0, (2.0/5.0)), (-5.0, 2.0, (2.0/5.0)), (np.array([5.0, -5.0, 0.5]), np.array([2.0, 2.0, 0.1]), np.array([(2.0/5.0), (2.0/5.0)], (1.0/5.0)))])
    def test_fractional_error(self, y_num, a_err, f_err):
        """Tests calc_error.fractional_error by asserting that calc_error.fractional_error(a_err, y_num) == f_err."""
        assert calc_error.fractional_error(a_err, y_num) == pytest.approx(f_err) # pytest.approx copes with floating point arithmetic issues

class FileTests:
    """Performs tests on save_error function and resulting files.
    """
    
    delete_after = True # True if files should be deleted after tests complete
    
    @staticmethod
    def _cleanup(paths):
        """Cleanup method to delete generated files after tests complete."""
        for p in paths: # For each path...
            with contextlib.suppress(FileNotFoundError, OSError):
                os.remove(p) # Delete the file, ignoring errors if the file does not exist or an OSError occurs (but not if p is a directory)
    
    @pytest.mark.parametrize("method,t,abs_err,frac_err,y", [("1DTest", np.array([]), np.array([]), np.array([]), np.array([])), ()])
    def test_save_error(self, method, t, abs_err, frac_err, y):
        """Tests the calc_error.save_error function.
               
            Tests:
                - Files save without raising exception.
                - Files exist afterwards.
                - Files can be loaded back into program by numpy.
                - Arrays from files have same length and shape as originals."""
        
        paths = calc_error.save_error(method, t, abs_err, frac_err, y) # Save the arrays to file (test will fail if any exception thrown)
        
        for p, arr in zip(paths, [t, abs_err.T, frac_err.T, y.T]): # For each path, get the path and the associated array (transposing the errors and y so they match the file contents)...
            assert Path(p).is_file() # Make sure that the path points to a file that exists
            
            saved_array = np.loadtxt(p) # Load the array from file
            assert len(saved_array) == len(arr) # Check that it has the same length as the original array
            assert np.shape(saved_array) == np.shape(arr) # Check that it has the same shape as the original array
        
        if self.delete_after: # If files should be deleted after completion of tests...
            self._cleanup(paths)