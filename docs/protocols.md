# Experimental protocols

The four configurations describe the historical method. Sample indices define every split, and the original CSV manifests remain unchanged. Acquisition-rate metadata explains the recordings; it does not trigger resampling.

## Recordings and splits

| Profile | Training | Validation | Test | Classes |
|---|---|---|---|---:|
| Cross-load | Loads 0/1, 580 windows | Load 2, 290 windows | Load 3, 290 windows | 10 |
| Cross-fault-size | .007-inch faults, 60 windows | .014-inch faults, 60 windows | .021-inch faults, 60 windows | 4 |
| STFT A/B | Indices 0–19, 200 windows | Indices 21–24, 40 windows | Indices 26–28, 30 windows | 10 |

Cross-load uses the first 29 segments of each recording. The ten class names, in order, are Healthy, IR_007, Ball_007, OR_007, IR_014, Ball_014, OR_014, IR_021, Ball_021, and OR_021. IR denotes inner race, Ball denotes rolling-element faults, and OR denotes outer race at the six-o'clock position.

The recording ranges below follow loads 0 through 3 in order.

| Condition | .007 inch | .014 inch | .021 inch |
|---|---|---|---|
| Inner race | 105–108 | 169–172 | 209–212 |
| Ball | 118–121 | 185–188 | 222–225 |
| Outer race | 130–133 | 197–200 | 234–237 |

Healthy records are 97–100. Cross-fault-size uses only load 0: fault records 105/118/130 for training, 169/185/197 for validation, and 209/222/234 for test, with indices 0–14. Class order is Healthy, IR, Ball, OR. Healthy record 97 supplies indices 0–14, 16–30, and 32–46 respectively. Indices 15 and 31 are omitted.

A/B uses the ten load-0 records and the ten-class mapping. Indices 20 and 25 are gaps. The same recordings appear in all three splits, so A/B validation is a representation exploration rather than independent-recording evaluation.

Independent fingerprints in `data/historical_reference.json` protect manifest bytes, scientific field assignments, class order, and historical image pixels. They were established from preserved evidence, not from each prospective dataset. Updating those references constitutes a protocol change requiring explicit review.

## From signal windows to pixels

Every STFT window contains 4096 samples. SciPy uses a 512-sample Hann window, a 512-point FFT, no boundary extension, no padding, and its default spectrum scaling and detrending. Profile A overlaps by 256 samples; the other profiles overlap by 448. Magnitudes become `20 * log10(abs(Z) + 1e-12)`, giving arrays of 257×15 for A and 257×57 otherwise. PNG width and height are therefore 15×257 or 57×257.

The .5 and 99.5 percentiles are fitted independently on concatenated training spectra. Values are clipped, normalized to 0–255, converted to uint8, and flipped vertically. A/B and cross-fault-size use float64 signals and truncate. Cross-load uses float32 signals and NumPy rounding before conversion. Separate scalar percentile calls preserve the float32 limits used by the historical cross-load implementation. Stored reference limits verify new computations; they are not substituted into generation.

No profile resamples. A healthy window lasts approximately 85.33 ms at 48 kHz, whereas a faulty window lasts 341.33 ms at 12 kHz. Relabeling the STFT rate does not harmonize the signal or its physical frequency axis.

## Model and optimization

ImageFolder decodes images to RGB. Cross-load/A/B convert to three-channel grayscale before resizing; cross-fault-size resizes before grayscale conversion. All resize to 224×224 using bilinear PIL interpolation with antialiasing, then use tensor conversion and ImageNet normalization. There is no augmentation.

EfficientNet-B0 starts with torchvision DEFAULT ImageNet weights. Its classifier is replaced by dropout .2 and a 1280-input linear layer with ten or four outputs. Stage two unfreezes the final projection, `features[-1]`, rather than an earlier MBConv stage.

| Stage | Epochs | Classifier learning rate | Projection learning rate | Training mode |
|---|---:|---:|---:|---|
| Stage 1, all profiles | 15 | 1e-3 | Frozen | Features eval, classifier train |
| Stage 2, cross-load | 10 | 1e-4 | 1e-5 | Earlier features eval, final projection train |
| Stage 2, cross-fault-size | 20 | 1e-3 | 5e-5 | Model train, all BatchNorm layers eval |

Optimization uses AdamW with weight decay 1e-4 and cross-entropy loss, without a scheduler or early stopping. Cross-fault-size leaves frozen stochastic layers active in stage two. Selection maximizes validation macro-F1, using lower loss to break ties. Cross-load/A/B use `np.isclose`; cross-fault-size uses exact equality. The tolerance can admit a slightly lower cross-load F1 when loss improves, matching the original selection rule.

Each stage restores its selected state. Before stage two, the runner performs the additional validation pass present in original one-based cells D45 and D65. This also preserves the relevant validation-iterator RNG consumption. Stage-one checkpoints remain independent of subsequent in-memory updates. Test evaluation never selects checkpoints.

## Random state and fidelity boundaries

Cross-load/A/B retain seed 42. A/B training loaders use dedicated seed-42 generators, and A consumes a sanity batch before model initialization. Cross-fault-size also consumes its original pre-model sanity batch (D58), without adding a seed reset. Cross-load and cross-fault-size loaders use global RNG state. Batch sizes are 16 for cross-load/A/B and 8 for cross-fault-size. Workers and pinning default to 2/True and 0/False respectively. Explicit worker overrides are recorded and may affect execution.

Cross-fault-size introduces no new seed because its original section inherited notebook state. Prospective runs capture Python, NumPy, CPU torch, available CUDA, and explicit loader-generator states before initialization and training, and at stage-two entry. These JSON records are passive observations, not a new determinism policy or exact mid-epoch resume system. CUDA state collection may initialize the CUDA runtime; it does not sample its generators.

The original ambient Colab state is unavailable. Separate A/B CLI runs also cannot reconstruct the complete shared A-then-B session. Hardware, library versions, earlier notebook activity, and worker behavior can change a new run. Capturing future state does not recover historical weights or guarantee historical scores.
