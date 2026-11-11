# C5 — readability fixes for Part C figures

Date: 2026-07-10 (fresh session, plotting-only task)

## What changed

Plotting code only (`scripts/make_figures.py`); no new runs, no record changes.
Sources remain the same: latest non-superseded C2 (python 3.12) ok rows in
`results/runs.jsonl` (see results/ANNOTATIONS.md).

### figures/precision_size_latency.png
- The MLP and GCN panels had no tick values (minor ticks off removed the only
  labels inside narrow ranges).
- Added a dense seed-0 FP32 reference point per panel (star marker, from the
  latest factorial `dense_fp32` row: artifact bytes + median latency).
- Every point is now labeled with its value: `format (bytes B, ms)`.
- Tick readability: axes spanning >= 1 decade use log scale with
  `LogLocator(subs=(1,2,5))` + `ScalarFormatter` plain-number labels and minor
  ticks off; axes under 1 decade (e.g. MLP x) use a linear scale.
- Latency unit/batch stated in the y axis label: `median latency (ms, batch_256)`
  for MLP/CNN, `median latency (ms, full_graph)` for GCN.
- New helper `_readable_ticks(ax)` asserts every axis has >= 2 visible major
  tick labels (applies to both figures).

### figures/factorial.png
- The bytes and latency rows had no tick values. Now:
  - every bar is labeled with its value (accuracy 4 decimals, bytes in KB,
    latency in ms with 3 significant digits);
  - bytes/latency y-axes are log with `LogLocator(subs=(1,2,5))` and
    plain-number formatters (bytes axis formatted in KB);
  - y-limits padded (`[min/1.1, max*1.8]`) so labels are not clipped;
    accuracy row uses a tight linear range.
- Layout tightened: shared condition legend dropped (x tick labels already name
  the four conditions) and `tight_layout` without the reserved legend band,
  removing the empty band above the grid.

### figures/CAPTIONS.md
- Both C2 captions updated to describe the new labels/ticks/reference point and
  dropped legend.
- Script now upserts its two caption sections (single copy, idempotent) instead
  of appending duplicates on every run.

## Verification
- `PYTHONPATH=code uv run python scripts/make_figures.py size-latency factorial`
  regenerates both PNGs; the >=2-tick-label assertion passes on every axis.
- `PYTHONPATH=code uv run pytest -q`: 153 passed.
