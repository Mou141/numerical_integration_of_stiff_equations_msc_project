"""Test suites for ../filter_initial_values.py for use with pytest."""

from pathlib import Path
import sys

import pytest

# Add this file's parent directory to the python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import filter_initial_values


class TestCmdArgs:
    """Tests that filter_initial_values.parse_args parses command line arguments correctly."""

    @pytest.mark.parametrize("args", [["file1.tsv", "file2.tsv"]])
    def test_successful_args(self, args):
        """Checks that the arguments passed to filter_initial_values.parse_args are returned correctly."""

        assert filter_initial_values.parse_args(args) == args

    # Arguments that should fail
    # Empty
    fail_args_1 = []
    # Only 1 argument
    fail_args_2 = ["test1.tsv"]
    # Too many arguments
    fail_args_3 = ["test1.tsv", "test2.tsv", "test3.tsv"]

    @pytest.mark.parametrize("args", [fail_args_1, fail_args_2, fail_args_3])
    def test_fail_args(self, args):
        """Checks that argparse attempts to exit the program when the specified arguments are passed to filter_initial_values.parse_args."""

        with pytest.raises(SystemExit):
            filter_initial_values.parse_args(args)
