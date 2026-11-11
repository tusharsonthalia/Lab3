Continue Task C1b from where you stopped. Your last reply was cut off after two tokens ("Now let"), so
no final report was produced. Current state I checked: results/precision_support.md exists; the C1 debug
files are in _quarantine/c1_debug/; tests/test_precision.py currently fails at collection (import or
syntax error); no phase="precision" or "factorial" rows exist yet in results/runs.jsonl.
Next: fix the test file collection error, then do the remaining C1b items in order: conversions + real
records for all formats and all 3 archs, reload-equivalence, INT8 selection, factorial, figures,
partc_summary.md, the Part B legend cleanup, then `uv run pytest -q` and the final report. Same rules.
