#!/usr/bin/env python3
"""Run the 2D/3D PDE grid-scaling validation."""

import json

from transfermod.validation import run_pde_scaling_validation

if __name__ == "__main__":
    result = run_pde_scaling_validation()
    print(json.dumps(result.to_dict(), indent=2))
