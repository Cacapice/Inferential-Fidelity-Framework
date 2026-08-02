#!/usr/bin/env python3
"""Run the trained-operator aggregate/decision coupling audit."""

import json

from transfermod.validation import run_learned_operator_validation

if __name__ == "__main__":
    result = run_learned_operator_validation()
    print(json.dumps(result.to_dict(), indent=2))
