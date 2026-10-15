# Installation and execution

Start by inspecting the notebooks without raw data. Generating images and training are separate, explicit steps. Use a source checkout: configurations and manifests are repository resources, and standalone wheel distribution is not supported.

## Environment choices

The inspected reference environment is Windows CPU with Python 3.13.2, NumPy 2.2.4, SciPy 1.15.2, pandas 2.2.3, Pillow 11.2.1, scikit-learn 1.6.1, torch 2.7.1+cpu, and torchvision 0.22.1+cpu. `requirements-inspected-cpu.txt` records its package versions. These are not a recovered Colab lock file. See [verification and reproducibility](validation_report.md) for actual installation and execution outcomes.

The package declares Python 3.10+ and broader dependency bounds. Those bounds express intended compatibility, not a tested environment matrix. GPU and Colab execution remain unverified. No hardware speed or memory estimate has been established for full training.

### Windows notebook inspection

The publication subset was installed and checked with notebook extras in a repository-local virtual environment without system-site packages. `requirements-notebooks-windows.txt` records the tested direct package versions. This environment does not include torch and needs no raw recordings for the safe notebook defaults.

From the checkout root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks-windows.txt
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
.\.venv\Scripts\python.exe -m jupyter lab --ServerApp.root_dir=. notebooks/01_signal_exploration.ipynb
```

The notebook runner checks a minimal calculation before attempting the five walkthroughs:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
```

It selects the executing interpreter, uses repository-local configuration and temporary directories, and uses the Windows selector event loop for ZeroMQ compatibility. Execution records and executed copies remain under ignored `work/notebook_execution/`. A startup failure stops the preflight rather than repeating the same failure for all five notebooks.

For optional CPU model work, the inspected torch/torchvision pair is installed separately:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

This pair follows the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). It was already available for CPU engineering tests; a fresh full ML installation and research training were not part of the isolated notebook checks. `requirements-inspected-cpu.txt` describes the older inspected environment, not the new notebook environment.

For Linux/macOS, create a virtual environment with `python3 -m venv .venv`, activate it using your shell, and install `.[notebooks]`. These platforms and interactive browser operation have not been verified here. Launch from the checkout root or its notebooks directory so setup can locate the public resources. The Windows automated checks use the same notebook cells through actual Jupyter kernels.

### Google Colab

Upload a checkout to the runtime and unpack it, or clone the actual repository once published. Change the notebook working directory to that checkout before running setup.

In a Colab setup cell, replace the example directory with the uploaded checkout:

```python
%cd /content/cwru-bearing-generalization
%pip install -e ".[notebooks]"
```

Inspect the runtime's torch and torchvision versions before enabling training. A compatible installation depends on the current runtime; do not replace its CUDA stack blindly. If using Drive for raw data or persistent outputs, mount it explicitly and set `CWRU_DATA_ROOT` and `CWRU_ARTIFACT_ROOT` to your own folders. `/content` is temporary. This setup is guidance, not a verified Colab execution claim.

## Raw inputs and artifact locations

Follow the [data guide](../data/README.md) to obtain the selected MATLAB recordings. The loader selects the exact `X<id>_DE_time` key, including X099 in record 99, rather than choosing the first matching array. Validation checks lengths, manifest bounds, and local-reference file hashes. Those hashes identify inspected files; they are not official CWRU checksums.

For optional notebook data access in PowerShell:

```powershell
$env:CWRU_DATA_ROOT = 'D:\CWRU\mat-files'
$env:CWRU_ARTIFACT_ROOT = "$PWD/runs"
```

The following CLI examples run from the checkout root in PowerShell using the same virtual environment. Replace `/your/mat-directory` and other example paths with your own paths. CLI commands use explicit paths and do not require these variables. Generated directories must be new; overwriting is rejected.

```powershell
.\.venv\Scripts\python.exe -m cwru_bearing.cli validate --profile cross_fault_size --data-root /your/mat-directory
.\.venv\Scripts\python.exe -m cwru_bearing.cli generate --profile cross_fault_size --data-root /your/mat-directory --output data/generated/cfs
```

Generation writes grayscale PNGs, `manifest.csv`, and schema-2 `preprocessing.json`. It computes train-only limits using the historical method without resampling. The full generation configuration is retained, while a separate preprocessing identity excludes runtime-only settings.

Training validation decodes every image and checks dimensions, mode, inventory, semantic manifest identity, metadata invariants, fitted limits, and independently preserved pixel fingerprints. Older metadata is rejected with an explicit regeneration requirement. Do not hand-edit old metadata to bypass validation. A historical-image comparison is available without writing new images:

```powershell
.\.venv\Scripts\python.exe -m cwru_bearing.cli compare --profile cross_fault_size --data-root /your/mat-directory --images /historical/image-root
```

## Training and evaluation

Training requires the explicit `--execute-training` flag and may download torchvision ImageNet weights on first use. The historical scores have not been independently reproduced by a new full training run.

```powershell
.\.venv\Scripts\python.exe -m cwru_bearing.cli train --profile cross_fault_size --images data/generated/cfs --output runs/cfs/run01 --execute-training
.\.venv\Scripts\python.exe -m cwru_bearing.cli evaluate --checkpoint runs/cfs/run01/stage2_best.pth --images data/generated/cfs --output runs/cfs/eval01
```

The same commands accept `stft_a`, `stft_b`, or `cross_load` where a profile is required. A/B run stage one only and do not produce historical test scores. Final protocols run two stages, restore validation-selected weights, and evaluate test. Only load trusted checkpoints.

For Windows notebook multiprocessing difficulties, prefer the terminal CLI. An explicit `--num-workers 0` override is available on `train`; the default configuration is otherwise retained. Worker changes do not invalidate identical preprocessing, but can affect stochastic execution and must be reported.

Runs retain full resolved configuration, class mapping, preprocessing metadata, source/input hashes, environment, JSON RNG snapshots, selected checkpoints, epoch histories, and image-linked predictions. Additional boundary validation and stage-two entry state are saved for two-stage runs. JSON RNG state can be restored with `restore_random_state` in a compatible environment; exact interrupted-training resume is not implemented because sampler position, process state, and full execution context are not captured.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_notebooks.py
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
```

The last command executes all five safe default paths through actual Jupyter kernels and writes only under ignored `work/`. It removes the raw-data environment variable for those runs. Full model checks require the optional torch/torchvision installation above; a notebook-only environment skips those checks. Tests use temporary synthetic inputs, random-weight or tiny models, and no pretrained downloads. Their metrics are engineering fixtures, not additional CWRU findings.

Git attributes preserve file bytes across platforms because historical manifests and recording metadata are validated using fixed SHA-256 fingerprints. Do not normalize the line endings of those provenance inputs.
