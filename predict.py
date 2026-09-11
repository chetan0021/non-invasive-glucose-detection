"""
PHASE 7 — Root Production Inference & Uncertainty Quantification Module

Usage:
    from predict import GlucosePredictor

    predictor = GlucosePredictor()
    result = predictor.predict(input_dict)

CLI Test:
    python predict.py --test
"""

import sys
from scripts.predict import GlucosePredictor, run_cli_tests

__all__ = ["GlucosePredictor", "run_cli_tests"]

if __name__ == "__main__":
    run_cli_tests()
