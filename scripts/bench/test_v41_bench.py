"""Reject CLI-success results that do not represent the requested workload."""

import unittest

from v41_bench import validate_result


class ResultValidationTest(unittest.TestCase):
    def setUp(self):
        self.result = {
            "completed": 4,
            "failed": 0,
            "total_output_tokens": 512,
            "output_throughput": 30.0,
            "total_token_throughput": 270.0,
            "median_ttft_ms": 700.0,
            "median_tpot_ms": 25.0,
        }

    def test_complete_workload(self):
        validate_result(self.result, 4, 128)

    def test_zero_exit_with_failed_requests_is_not_a_benchmark(self):
        for changes in (
            {"completed": 0, "failed": 4},
            {"failed": 1},
            {"total_output_tokens": 511},
            {"median_tpot_ms": float("nan")},
            {"output_throughput": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_result(self.result | changes, 4, 128)


if __name__ == "__main__":
    unittest.main()
