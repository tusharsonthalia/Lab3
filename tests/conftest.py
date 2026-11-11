"""Pytest configuration.

P2 (A2 review): pytest must NEVER write to the real results/runs.jsonl.
The results path is configurable via the LAB3_RESULTS_DIR environment
variable (see lab3.util.RESULTS_DIR); we point it at a session tmp dir
BEFORE the lab3 package is imported, because util reads the env var at
import time. Every record the suite appends lands in the tmp dir only.
"""
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Must happen before any `import lab3...`: util.RESULTS_DIR is fixed at import.
# Must happen before any `import lab3...`: util.RESULTS_DIR and lab3.prune's
# PILOT_DIR (Part B arm-checkpoint root) are fixed at import. The pilot dir
# defaults to checkpoints/pilot (sibling of the read-only checkpoints/dense
# parent root); redirecting it here keeps pytest out of the real tree.
_TMP_RESULTS = os.path.join(tempfile.mkdtemp(prefix="lab3_pytest_results_"), "results")
os.environ["LAB3_RESULTS_DIR"] = _TMP_RESULTS
os.environ.setdefault(
    "LAB3_PILOT_DIR",
    os.path.join(tempfile.mkdtemp(prefix="lab3_pytest_pilot_"), "pilot"),
)

sys.path.insert(0, str(_REPO / "code"))
