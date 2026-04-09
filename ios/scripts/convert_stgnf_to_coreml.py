#!/usr/bin/env python3
"""
Run the repository CoreML conversion script from the iOS workspace.

This wrapper keeps the feature-local task path stable while delegating to the
authoritative converter in the repo root, which is configured to use the
Apr01_1416 Multi-run checkpoint for the iOS app.
"""

from pathlib import Path
import runpy

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_stgnf_to_coreml.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
