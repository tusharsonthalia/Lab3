Task B3b — the three Part B figures are wrong; fix the plotting code only (no new runs, no changes to
results/runs.jsonl or any CSV values). Read figures/CAPTIONS.md, results/pilot_summary.csv and
results/confirm_summary.csv first, and find the plotting code that produced the figures.

I inspected the PNGs:
1. figures/latency_vs_accuracy.png is EMPTY — all three panels have axes but no points. Most likely the
   latency columns are read under the wrong names or are NaN after parsing. Plot every pilot
   post_finetune arm: x = median latency (batch 256 for mlp/cnn, full_graph for gcn, log scale), y = test
   accuracy; for each arm one hollow marker for dense_zeros and one filled marker for compact; marker
   shape = method; dense seed-0 baseline as a black star. Assert in the script that each panel has at
   least 32 plotted points (16 arms x 2 paths) and fail loudly otherwise.
2. figures/size_vs_sparsity.png: the x axis is an index 0..3 instead of achieved sparsity, and a series
   whose values are < 1 (sparsity fractions) is plotted as bytes (the curves near 10^0). Fix: x =
   achieved eligible-weight sparsity (0..1), y = bytes on disk (log), series = for each method its
   dense_zeros, gzip, and compact bytes (color = method, linestyle = artifact type). Assert all y values
   > 1000 bytes.
3. figures/acc_vs_sparsity.png has no legend, and the reference-line text labels overlap. Add a legend
   (method colors; dashed = post_prune, solid = post_finetune; open markers = confirm mean ± std) and
   move reference labels into the legend instead of on-plot text.
All three: one shared legend per figure (outside the panels or in an empty corner), titles per panel,
dpi 150, readable at memo size (font >= 8 pt). Update figures/CAPTIONS.md. Run `uv run pytest -q`.
Report what was wrong in each and attach nothing else.
