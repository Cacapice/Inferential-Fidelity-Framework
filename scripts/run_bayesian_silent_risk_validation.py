#!/usr/bin/env python3
"""Run the posterior Silent Risk validation."""

import json

from transfermod.validation import run_bayesian_silent_risk_validation

if __name__ == "__main__":
    result = run_bayesian_silent_risk_validation()
    print(json.dumps(result.to_dict(), indent=2))
