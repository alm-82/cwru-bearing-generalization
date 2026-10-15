# Data acquisition and validation

Raw files and generated datasets are not included. Obtain normal load-0/1/2/3 files from https://engineering.case.edu/bearingdatacenter/normal-baseline-data and the .007/.014/.021-inch DE fault files at loads 0–3 from https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data. Use outer-race faults centered at 6:00. The linked MATLAB files have the record IDs below; save them as `<id>.mat` (for example, `105.mat`) in an external folder.

Required IDs for cross-load: 97, 98, 99, 100, 105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133, 169, 170, 171, 172, 185, 186, 187, 188, 197, 198, 199, 200, 209, 210, 211, 212, 222, 223, 224, 225, 234, 235, 236, 237.

CFS and A/B need only 97,105,118,130,169,185,197,209,222,234. Use the original MATLAB arrays, not renamed/reduced third-party conversions. 99 contains duplicate X098 arrays; the loader deliberately selects X099_DE_time.

recordings.csv contains acquisition rate (documented), historical processing rate, expected keys, sample counts, embedded RPM where present, local-reference SHA-256 and quality flags. Healthy rates are sourced from Smith & Randall Table A1, not inferred from signal lengths. Fault rate comes from the official 12k DE collection. Checksums fingerprint the inspected local files; they are not official distribution certificates. If a download differs, investigate instead of editing metadata automatically.

Manifests are original byte-identical CSVs. data.load_manifest normalizes IDs in memory and validates exact fields and nonoverlap. Do not relocate or overwrite raw recordings from within this repository. 

`historical_reference.json` stores independently established manifest, semantic, recording-metadata, and decoded-pixel fingerprints. The supported validator compares against these fixed references rather than rereading an input as its own authority. Only fingerprints are distributed; neither raw signals nor generated image datasets are included.
