"""Tests for the modal evaluation pipeline script."""

import unittest

import numpy as np

from scripts.modal_eval_pipeline import compute_modal_scores, evaluate_modal_metrics


class TestModalEvalPipeline(unittest.TestCase):
    """Test cases for modal scoring helpers."""

    def test_compute_modal_scores_normalizes_to_unit_interval(self):
        """Normalized score stays in [0, 1] and legacy score maps to [1, 5]."""
        scores = compute_modal_scores(1.25)
        self.assertAlmostEqual(scores["auto_score_0_1"], 1.0)
        self.assertAlmostEqual(scores["auto_score_1_5"], 5.0)

        scores = compute_modal_scores(-0.25)
        self.assertAlmostEqual(scores["auto_score_0_1"], 0.0)
        self.assertAlmostEqual(scores["auto_score_1_5"], 1.0)

    def test_evaluate_modal_metrics_exposes_normalized_score(self):
        """Metric output includes normalized and legacy score fields."""
        chroma = np.zeros((12, 4), dtype=np.float32)
        chroma[2, :] = 0.7
        chroma[4, :] = 0.3

        metrics = evaluate_modal_metrics(chroma, mode="dorian")

        self.assertIn("auto_score_0_1", metrics)
        self.assertIn("auto_score_1_5", metrics)
        self.assertGreaterEqual(metrics["auto_score_0_1"], 0.0)
        self.assertLessEqual(metrics["auto_score_0_1"], 1.0)
        self.assertGreaterEqual(metrics["auto_score_1_5"], 1.0)
        self.assertLessEqual(metrics["auto_score_1_5"], 5.0)
        self.assertAlmostEqual(
            metrics["auto_score_1_5"], 1.0 + 4.0 * metrics["auto_score_0_1"]
        )


if __name__ == "__main__":
    unittest.main()
