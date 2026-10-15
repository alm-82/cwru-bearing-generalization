# Evidence-to-implementation map

Cell indices are one-based notebook positions, counting Markdown/empty cells, not execution counts. E=01_cwru_exploration.ipynb; D=CWRU.ipynb. Original working notebooks are retained unchanged privately. Public evidence consists of selected output logs, matrices, figures, and this map; the full working notebooks are not distributed. Public extracts support inspection of the selected observations; references to other original cells identify private source material and cannot be followed within this checkout.

| Original cells | Extracted component | Supported notebook / evidence |
|---|---|---|
| E49/E51/E76/E81 | classical.windows/features/svm_pipeline | 02; E82/E85 in the [classical evidence extract](../results/historical/classical_svm_evidence.json) |
| E100–103 | Fault-only transfer interpretation | 02; E102/E103 in the public classical evidence extract; no replacement run performed |
| E8–18/E34–43/E108–139 | FFT, envelope, bearing targets, harmonic rules and spacing candidates | 01; [method audit](exploration.md) and [public source/output extracts](../results/historical/signal_exploration_evidence.json); rules remain exploratory |
| E148/E150/E154 | preserved manifest.csv; preprocessing profiles A/B | 03; D23/D25 archived logs |
| E163–175 | cross_load config, data loader, preprocessing | 04; D50–52 historical result |
| E185/E186 | cross_fault_size config and preprocessing | 05; D79–80 historical result |
| D7/D8/D42/D57 | models.build_loaders, transforms | 03/04/05; protocol-specific orders and generators |
| D14/D42/D59 | models.build_model | 03/04/05; 10/4-class classifier |
| D42/D43/D46/D47 | training.train_stage and stage modes | 04 |
| D63/D67/D70/D71 | training.train_stage and stage modes | 05 |
| D10/D58 | training.run_training pre-model loader iteration | 03/05; preserves the original sanity-batch events without adding a CFS seed |
| D45/D65 | training.train_stages boundary validation and iterator RNG | 04/05; preserves the original validation pass between training stages |
| D50/D79 | metrics and historical matrices | figures and results/historical |

New persistence and path configuration are engineering additions. They do not recover missing weights or imply new scientific results. Exact session RNG history remains a fidelity boundary. Scalar percentile calls retain the historical precision; all 1,880 reference image arrays match exactly.

The joint load/size SVM investigation (E86–95) is also restored in notebook 02 and `classical.run_baselines`, with E93/E95 stdout in the public classical extract. It is distinct from fault-only size transfer.
