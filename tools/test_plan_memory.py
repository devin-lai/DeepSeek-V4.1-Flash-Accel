"""The expert offload budget excludes scales, and spills whole parameters."""

import unittest

from plan_memory import GIB, pinned_offload_bytes


class PinnedBudgetTest(unittest.TestCase):
    config = {
        "num_hidden_layers": 40,
        "n_routed_experts": 384,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
    }

    def test_reference_12_gib_budget_counts_payload_only(self):
        stock, _ = pinned_offload_bytes(self.config, 8, True, 4.25, 12 * GIB)
        exact, _ = pinned_offload_bytes(self.config, 8, True, 4.25, 12 * GIB, True)
        # Measured allocation: 16 w13 buffers + 15 w2 buffers per rank.
        self.assertEqual(stock / GIB, 23.5)
        self.assertEqual(exact / GIB, 12.392578125)

    def test_smaller_budget_still_spills_the_last_whole_parameter(self):
        exact, _ = pinned_offload_bytes(self.config, 8, True, 4.25, 9 * GIB, True)
        self.assertEqual(exact / GIB, 9.228515625)
        self.assertEqual(
            pinned_offload_bytes(self.config, 8, True, 4.25, 0, True)[0], 0
        )


if __name__ == "__main__":
    unittest.main()
