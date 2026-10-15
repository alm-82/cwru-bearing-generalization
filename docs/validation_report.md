# Verification and reproducibility

Checks rerun on 15 October 2026 distinguish preservation of historical evidence from new execution of the documented methods. No full EfficientNet training run has independently reproduced the historical scores.

| Check | Evidence |
|---|---|
| Historical configurations, manifests, recording metadata, result JSON and seven saved logs | Preserved without numerical changes |
| Recording validation | All 40 selected records match the stored local checksums, expected signal keys, sample counts, and segment bounds |
| Historical preprocessing | All 1,880 spectrogram arrays match exactly across four profiles; zero changed pixels |
| Engineering tests | 44 CPU tests pass without skips in the inspected ML environment |
| Model semantics | Classifier dimensions, trainable parameter counts, BatchNorm/stochastic-layer modes, checkpoint selection and restoration checked |
| Random-state fidelity | Cross-fault-size pre-model sanity iteration and stage-boundary validation order covered by regression tests; no CFS seed reset added |
| Classical historical evidence | Six verbatim saved outputs match E82/E85/E93/E95/E102/E103; structured accuracies and confusion matrices agree, including RMS-only transfer 231/708 |
| Signal-processing methods | Eleven synthetic/source-equivalence tests cover raw/Hann FFT, filtering, Hilbert envelopes, historical frequency coefficients, harmonic scores, rules and shaft-spacing search |
| Classical methods | Synthetic tests check feature extraction, training-only scaling, selected recording families, and split sample counts for the three SVM protocols |
| Public notebooks | All five schemas and code-cell syntax pass; 33 code cells execute through genuine Jupyter kernels, following a successful minimal preflight |
| Saved notebook outputs | Tables and synthetic illustrations come from safe execution; final research scores are read from historical evidence |
| Publication boundary | Tests reject path escapes, unexpected candidate files, private-content patterns, error outputs, and existing destinations |

The complete 44-test suite was rerun in the inspected CPU ML environment with this checkout on the import path. All five default notebooks were separately rerun in fresh Jupyter kernels using the repository-local virtual environment. A fresh PowerShell session verified that environment, checked its installed dependencies, validated all 40 recordings, and generated the 180 CFS images in an ignored directory. Existing environments were reused; no fresh dependency installation or full training was performed.

The full CPU suite uses temporary synthetic datasets, random-weight models, and a tiny offline training/checkpoint/evaluation round trip. Its metrics are test fixtures, not additional CWRU findings. The separate Windows notebook environment uses Python 3.13.2 and the direct versions in `requirements-notebooks-windows.txt`; it does not require torch or raw recordings for the default notebook paths.

The notebook runner removes raw-data access for safe execution and uses a Windows selector event loop for kernel compatibility. Local Jupyter client/kernel connections must be permitted. Successful cell execution and process exit are checked separately. Runtime diagnostics and executed verification copies remain under ignored `work/`.

Historical image comparisons read raw recordings without modifying them. The historical float32/scalar-percentile and float64/truncation differences remain intact, and no resampling or acquisition-rate harmonization is performed. Local recording hashes identify the inspected files; they are not official distribution checksums.

Full research training, historical-weight recovery, GPU execution, Colab, macOS, a broad dependency matrix, and exact interrupted-training resume are unverified. Hosted Linux CI results must be checked separately after any future publication. The original interactive Colab RNG history is unavailable; preserving known iteration events cannot recover it.

See [experimental protocols](protocols.md), [scientific limitations](scientific_audit.md), and [installation and execution](reproduction.md) for the scope of the maintained methods.
