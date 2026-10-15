# Interpreting the experiments

## Acquisition and provenance

The official CWRU 12k DE table identifies the selected faulty recording family. Its current normal-baseline page lists loads/RPM but no explicit sampling rate. Smith & Randall (2015), Appendix A Table A1, explicitly lists records 97–100 as 48 kHz normal data; Table A2 lists the 12 kHz DE faults. Local MATLAB files have expected keys and historical headers but no explicit rate field. The metadata therefore records a documented rate, not a rate measured from the file itself.

Local SHA-256 values fingerprint the inspected files. They are not authoritative CWRU checksums or proof of chain of custody. `99.mat` contains X098 DE/FE duplicates matching local 98; this anomaly is independently described in the [CWRU_Bearing_NumPy maintainer's changes](https://github.com/srigas/CWRU_Bearing_NumPy#changes). Its original cause is unknown. The loader selects X099 explicitly.

Smith & Randall Table 3 flags clipping in 236DE/237DE. Both participate in cross-load validation/test. Read-only local inspection found matching extremes within selected windows, but a simple plateau count did not determine the affected interval or model impact. These records remain in the historical protocol.

## Interpretation

Cross-load recording separation is valid, but physical bearing independence is not established. Ten class labels encode both fault type and size. The result concerns a particular held-out operating condition on this rig, not arbitrary unseen bearings or field deployment.

In cross-fault-size evaluation, faulty recordings differ across splits, while healthy training, validation, and test windows share recording 97. Nonoverlap and gaps remove direct window overlap, not shared-recording dependence. Selection already uses .014-inch faults. The .021-inch test is a specific transfer case, not universal size invariance.

Train-only pixel limits avoid direct normalization leakage. Healthy acquisition bandwidth/duration and class-dependent intensity remain possible shortcuts. The saved post-test intensity analysis does not prove the CNN used that shortcut. Separate ablations would be required.

The original CFS notebook inspected test-image statistics before evaluation and intensity afterward. No test-driven checkpoint selection was demonstrated; a fully untouched historical test workflow cannot be established from saved cells alone.

Segments from a recording are correlated. Per-window scores do not establish specimen-level uncertainty; binomial confidence intervals treating windows as independent are not included.

## Missing artifacts and prospective work

Original weights and standalone histories are unavailable. Archived logs are evidence, not independently reproduced predictions. No project-trained model is supplied. Prospective runs now preserve available random states, but these cannot recover the missing historical session. Domain adaptation and external-dataset evaluation have not been implemented.

Downsampling record 97 to 12 kHz would leave about 60985 samples, or 14 full 4096-sample windows. Historical cross-load requests 29, and CFS healthy portions run through index 46. A corrected pipeline therefore needs a separately defined and evaluated protocol, not a hidden substitution.

References: [official normal table](https://engineering.case.edu/bearingdatacenter/normal-baseline-data), [official fault table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data), [bearing geometry](https://engineering.case.edu/bearingdatacenter/bearing-information), [Smith & Randall DOI](https://doi.org/10.1016/j.ymssp.2015.04.021).
