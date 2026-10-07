#!/usr/bin/env python3
"""Master Test Suite Runner for Clickbait Detection & Categorization System.
Executes 4-Tier test suite across all project milestones.

Usage:
    python3 tests/run_all_tests.py
    python3 tests/run_all_tests.py --tier 1
    python3 tests/run_all_tests.py --milestone m1
    python3 tests/run_all_tests.py --verbose
"""

import argparse
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Clickbait Detection Test Suite (Tiers 1-4)")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Filter execution to a specific testing tier (default: all)")
    parser.add_argument("--milestone", choices=["m1", "m2", "m3", "m4", "e2e", "acceptance", "all"], default="all",
                        help="Filter execution to a specific milestone (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output (-vv)")
    parser.add_argument("-k", "--expression", type=str, default="",
                        help="Run tests matching given keyword expression")
    return parser.parse_args()


def build_pytest_args(args):
    pytest_args = ["-q"]
    if args.verbose:
        pytest_args = ["-v", "-s"]

    # Target files mapping
    if args.milestone == "m1":
        pytest_args.append(str(TESTS_DIR / "test_models.py"))
    elif args.milestone == "m2":
        pytest_args.append(str(TESTS_DIR / "test_api.py"))
    elif args.milestone == "m3":
        pytest_args.append(str(TESTS_DIR / "test_extension.py"))
    elif args.milestone == "m4":
        pytest_args.append(str(TESTS_DIR / "test_docs.py"))
    elif args.milestone == "acceptance":
        pytest_args.append(str(TESTS_DIR / "test_acceptance.py"))
    elif args.milestone == "e2e":
        pytest_args.append(str(TESTS_DIR / "test_e2e.py"))
    else:
        # Run all test files
        pytest_args.append(str(TESTS_DIR))

    # Tier filtering via expression if specified
    expressions = []
    if args.expression:
        expressions.append(args.expression)

    if args.tier == "1":
        expressions.append("Tier1 or Acceptance")
    elif args.tier == "2":
        expressions.append("Tier2 or Boundary")
    elif args.tier == "3":
        expressions.append("Tier3 or CrossFeature")
    elif args.tier == "4":
        expressions.append("Tier4 or RealWorld")

    if expressions:
        pytest_args.extend(["-k", " and ".join(f"({e})" for e in expressions)])

    return pytest_args


def main():
    args = parse_args()

    print("=" * 72)
    print(" 🧪 CLICKBAIT DETECTION & CATEGORIZATION SYSTEM — TEST SUITE RUNNER")
    print("=" * 72)
    print(f" Working Directory : {PROJECT_ROOT}")
    print(f" Target Milestone  : {args.milestone.upper()}")
    print(f" Target Tier       : {args.tier.upper()}")
    print("-" * 72)

    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    pytest_args = build_pytest_args(args)

    print(f" Invoking pytest with: {' '.join(pytest_args)}")
    print("-" * 72)

    # Run pytest and capture return code
    exit_code = pytest.main(pytest_args)

    print("-" * 72)
    if exit_code == 0:
        print(" ✅ TEST RUN STATUS: ALL TESTS PASSED OR CLEANLY SKIPPED")
    elif exit_code == 5:
        print(" ℹ️ TEST RUN STATUS: NO TESTS MATCHED FILTER CRITERIA")
        exit_code = 0
    else:
        print(f" ❌ TEST RUN STATUS: FAILURES DETECTED (Exit code: {exit_code})")
    print("=" * 72)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
