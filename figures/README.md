# Figure provenance

historical_confusion_matrices.png and .svg are rendered from results/historical/results.json, transcribed from D50 and D79. Rendering is presentation of existing evidence, not new inference. Rows are true labels, columns predicted labels. H=Healthy; IR=inner race; B=ball; OR=outer race. Suffix 7/14/21 means .007/.014/.021 inch. Different matrix sizes reflect different protocols and should not imply a controlled difficulty comparison.

scripts/make_figures.py regenerates the figure.

The cross-load matrix contains 287 correct predictions out of 290. The cross-fault-size matrix contains 42 out of 60; its prominent inner-race-to-outer-race errors are discussed in notebook 05.
