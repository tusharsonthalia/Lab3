Continue Task C1b. Your previous reply was cut off again ("For static CNN, `conv1`"). State I checked:
precision.py now uses native quantize_dynamic and FX static (good); tests/test_precision.py passes
(43 tests); there are still NO real phase="precision"/"factorial" rows in results/runs.jsonl, and no
int8_selection.json, factorial.csv, partc_summary.md or Part C figures.
Do the remaining items in this order, writing real records as you go:
 (1) standalone conversions for all formats x all 3 archs (fp16, bf16, dynamic INT8, static INT8, INT4
     simulated) with the full accounting fields and reload-equivalence; unsupported -> classified row;
 (2) results/int8_selection.json by the frozen C.1 rule; (3) the 12-condition factorial +
     results/factorial.csv; (4) Part C figures + CAPTIONS.md; (5) results/partc_summary.md;
 (6) the Part B legend cleanup in scripts/make_figures.py; (7) `uv run pytest -q`.
Because your replies keep getting cut off, keep each reply short and act through tool calls. When
everything is complete and tests pass, write results/C1b_DONE.md containing your full final report.
