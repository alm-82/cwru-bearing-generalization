# Historical evidence

These files preserve selected observations from privately retained original notebooks. Cell numbers are one-based positions, including Markdown and empty cells, not execution counts. E denotes `01_cwru_exploration.ipynb`; D denotes `CWRU.ipynb`. The complete working notebooks and original trained weights are not distributed.

| Public evidence | Original source | Contents |
|---|---|---|
| [Classical SVM extract](classical_svm_evidence.json) | E82, E85, E93, E95, E102, E103 | Verbatim saved outputs for four-feature and RMS-only SVMs across three protocols, with exact accuracy values, confusion matrices, feature sets, and class order. The original notebook SHA-256 identifies the private source. |
| [Signal exploration](signal_exploration_evidence.json) | Selected E8–139 and E183–184 | Verbatim formulas and saved observations; the E128 band-rule outcome is descriptive on inspected faults, not independent validation. |
| [A history](D_cell_023.txt), [B history](D_cell_025.txt) | D23, D25 | STFT A/B training and validation summaries. |
| [Cross-load stage one](D_cell_043.txt) | D43 | Training and validation history at printed precision. |
| [Cross-load evaluation](D_cell_050.txt) | D50 | Final historical test metrics and confusion matrix. |
| [CFS stage one](D_cell_064.txt), [CFS stage two](D_cell_072.txt) | D64, D72 | Training and validation histories, including weaker validation performance. |
| [CFS evaluation](D_cell_079.txt) | D79 | Final historical test metrics and confusion matrix. |
| [Structured results](results.json) | The outputs above | Historical observations, matrix-derived accuracy and macro-F1, and separately identified losses transcribed from saved logs. |

Text logs are verbatim stream extracts, not recovered full-precision training-history CSVs. Matrix-derived metrics do not recover per-image predictions or model weights. These records contain no newly trained model results.

New experiments write to separate ignored run directories. See [verification and reproducibility](../../docs/validation_report.md) for preprocessing and engineering checks.
