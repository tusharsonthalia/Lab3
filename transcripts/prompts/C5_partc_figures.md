Task C5 — readability fixes for two Part C figures (plotting code only; no new runs, no record changes).
Fresh session. Use `PYTHONPATH=code uv run ...`. Keep replies short; act through tool calls.

I inspected the PNGs:
1. figures/precision_size_latency.png: the MLP and GCN panels show NO tick values on either log axis
   (turning minor ticks off removed the only labels inside these narrow ranges). Add an FP32 reference
   point per panel (the dense seed-0 parent's artifact bytes and latency from the latest factorial
   dense_fp32 row), label every point with its value, and use readable major ticks with plain-number
   formatting (e.g. matplotlib.ticker.LogLocator(subs=(1,2,5)) + ScalarFormatter / FuncFormatter, or a
   linear axis if the range is under 1 decade). State the latency unit (batch 256 for mlp/cnn, full graph
   for gcn) in the axis label.
2. figures/factorial.png: the bytes and latency rows have no tick values, so the bars cannot be read, and
   there is a large empty band above the legend. Put the numeric value on top of every bar (bytes in KB,
   latency in ms with 3 significant digits, accuracy with 4 decimals), use readable tick labels, and
   tighten the layout (legend directly under the grid, or drop it since x tick labels already name the
   conditions).
Keep sources the same (latest non-superseded records; see results/ANNOTATIONS.md). Assert every axis has
at least 2 visible tick labels. Update figures/CAPTIONS.md if needed. Run `PYTHONPATH=code uv run pytest -q`.
When done write results/C5_DONE.md (what changed) and stop.
