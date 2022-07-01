"""pytest tests for ../analyse_error.py."""

# Add the parent directory of this file's directory to the python path (because analyse_error.py is in that directory)
import contextlib
from pathlib import Path # For file path manipulations
import sys # For sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest # For testing functions

import numpy as np # For maths and arrays

import os # For os.remove, to delete files
import random # To generate a random integer

import analyse_error # Module to test

class TestStatsFunctions:
    """Tests the functions which calculate statistics (i.e. l^2 norm, l^infinity norm, and mean)."""
    
    # Test parameters for test_2_norm
    l2_1 = (np.array([1.0, 1.0, 1.0, 1.0]), 2.0) # 1^2 = 1, sum is 4, sqrt(4) == 2
    l2_2 = (np.array([2.0, 2.0]), np.sqrt(8.0)) # 2^2 = 4, sum is 8
    l2_3 = (np.array([5.0, 1.5, 3.22, 5.6]), np.sqrt(68.9784)) # Sum is 68.9784
    l2_4 = (np.array([-2.0, 2.0]), np.sqrt(8.0)) # 2^2 == 4, (-2)^2 == 4, sum is 8 (i.e. check that negative numbers are squared correctly)

    @pytest.mark.parametrize("arr,l2", [l2_1, l2_2, l2_3, l2_4])
    def test_l_2_norm(self, arr, l2):
        """Tests that the l_2 norm of arr matches l2."""
        assert analyse_error.l_2_norm(arr) == pytest.approx(l2)
    
    # Test parameters for test_l_infinity_norm
    l_inf_1 = (np.array([2.0, 3.5, -1.0, 5.4]), 5.4) # 5.4 is the largest value so it should be returned
    l_inf_2 = (np.array([-6.5, 4.3, -3.0, 2.22]), 6.5) # 4.3 is the largest value, but -6.5 has the largest magnitude, so 6.5 should be returned

    @pytest.mark.parametrize("arr,l_inf", [l_inf_1, l_inf_2])
    def test_l_infinity_norm(self, arr, l_inf):
        """Tests that the l^infinity norm of arr matches l_inf."""
        assert analyse_error.l_infinity_norm(arr) == pytest.approx(l_inf)

    def test_l_infinity_norm_empty_array(self):
        """Tests that analyse_error.l_infinity_norm raises a ValueError when called with an empty array (np.max always raises a ValueError on an empty array)."""

        with pytest.raises(ValueError):
            analyse_error.l_infinity_norm(np.array([]))
    
    # Test parameters for test_calc_err_stats
    stats_1 = (np.array([2.0, -2.0]), (np.sqrt(8.0), 2.0, 0.0)) # l2 is sqrt(8), l_inf is 2, mean is 0
    stats_2 = (np.array([3.5, 12.3, -33.4]), (np.sqrt(1279.1), 33.4, -5.866666666666666))

    @pytest.mark.parametrize("arr,stats", [stats_1, stats_2])
    def test_calc_err_stats(self, arr, stats):
        """Tests that analyse_error.test_calc_err_stats matches the contents of "stats"."""

        stats = analyse_error.ErrorStatsTuple._make(stats) # Convert stats to an ErrorStatsTuple

        calc_stats = analyse_error.calc_err_stats(arr) # Calculate the statistics on the array

        # Compare each of the statistical measures in turn
        assert calc_stats.l2 == pytest.approx(stats.l2)
        assert calc_stats.l_inf == pytest.approx(stats.l_inf)
        assert calc_stats.mean == pytest.approx(stats.mean)

class TestCmdArgs:
    """Tests the code which parses the command line arguments."""
    
    # Test arguments for test_args
    args_1 = (["test.tsv"], ("test.tsv", None)) # The graph file should be None if no path is specified for it
    args_2 = (["test.tsv", "--graph-file", "test.png"], ("test.tsv", "test.png")) # Both data file and graph file path should be returned as input

    @pytest.mark.parametrize("args,out", [args_1, args_2])
    def test_args(self, args, out):
        """Asserts that the result of parsing 'args' with analyse_error.parse_cmd_args is equal to out."""
        assert analyse_error.parse_cmd_args(args) == out
    
    # Test arguments for test_fail_args
    fail_args_1 = [] # Empty array should fail as the data file is required
    fail_args_2 = ["test.tsv", "--graph-file", "test.png", "fgdgdg"] # Array with 3 arguments should fail as this is too many
    fail_args_3 = ["test.csv", "-dfgd", "dgdfg"] # Incorrect switch 
    fail_args_4 = ["test.csv", "--graph-file"] # Switch given but no file specified

    @pytest.mark.parametrize("args", [fail_args_1, fail_args_2, fail_args_3, fail_args_4])
    def test_fail_args(self, args):
        """Checks that argparse fails for incorrect arguments."""
        with pytest.raises(SystemExit):
            analyse_error.parse_cmd_args(args)

class TestDataFileRead:
    """Tests the code which reads absolute error data from a file."""

    @pytest.fixture(scope="function")
    def data_file(self, tmp_path):
        """Returns a temporary file name with the extension .tsv, and then attempts to delete it after the test is complete."""
        basename = "tmp_{0}.tsv".format(random.randint(0, 1000000)) # Generate a base filename of the format tmp_N.tsv, where N is a random integer between 0 and 100000

        full_path = tmp_path / basename # Create a full file path from the path of the temporary directory and the basename

        yield full_path # Return it

        # After test function is complete, attempt to delete file, ignoring missing file or IO problem
        with contextlib.suppress(FileNotFoundError, OSError):
            os.remove(full_path)
    
    # Test data for test_read_data_file
    arr_1 = np.array([np.array([3.3, 545.43, 6766.76, 8.887])]) # Test error data for 1D problem
    arr_2 = np.array([[5.654, 56767.65, 8787.6, 64.23], [76.66666, 232.232, 1232.32, 8757566.00]]) # Test error data for 1D problem
    arr_3 = np.array([np.array([np.nan, 6.5454, 2323.2])]) # Test that program processes NaN correctly

    @pytest.mark.parametrize("arr", [arr_1, arr_2, arr_3])
    def test_read_data_file(self, arr, data_file):
        """Creates a temporary data file, stores 'arr' in it, and then uses analyse_error.read_data_file to read it."""
        np.savetxt(data_file, arr.T, delimiter="\t") # Write the array to file, transposing it

        data = analyse_error.read_data_file(data_file) # Read the data back

        assert data == pytest.approx(arr, nan_ok=True) # Check that the data is the same after reading
        assert len(data.shape) == 2 # Check that the array is 2D
        assert data.shape == arr.shape # Check that the data and the original array are the same shape

class TestHistogram:
    """Tests the code which generates histograms from error data."""

    @pytest.fixture(scope="function")
    def data_file(self, tmp_path):
        """Returns a temporary file name with the extension .png, an then attempts to delete it after the test is complete."""
        basename = "tmp_{0}.png".format(random.randint(0, 1000000)) # Generate a base filename of the formar tmp_N.png, where N is a random integer between 0 and 1000000

        full_path = tmp_path / basename # Create a full file path from the path of the temporary directory and the basename

        yield full_path # Return it

        # After test function is complete, attempt to delete file, ignoring missing file or IO problem
        with contextlib.suppress(FileNotFoundError, OSError):
            os.remove(full_path)
    
    # Test data for test_histogram_file and test_histogram_display (random data with a standard distribution)
    data_1 = np.array([np.random.standard_normal(1000)]) # 1D data
    data_2 = np.array([np.random.standard_normal(1000), np.random.standard_normal(1000)]) # 2D data

    @pytest.mark.parametrize("data", [data_1, data_2])
    def test_histogram_file(self, data, data_file):
        """Creates a histogram from the specified data, writes it to file, and then checks that the file was written."""

        analyse_error.make_histogram(data, file_path=str(data_file))
        
        assert data_file.is_file() # Check that the file exists
    
    @pytest.mark.parametrize("data", [data_1, data_2])
    def test_histogram_display(self, data):
        """Creates a histogram from the specified data and displays it visually."""
        analyse_error.make_histogram(data)
        