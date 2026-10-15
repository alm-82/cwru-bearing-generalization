# Third-party attribution and scope

The MIT license covers original repository code and other material that Ali Mirzaabolhassani is entitled to license. It does not relicense third-party datasets, model weights, publications, software, or notices preserved in historical notebook output.

The vibration recordings originate from the [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). No raw recordings or dataset archives are included. Follow the data provider's terms when obtaining or redistributing recordings. Local checksums and manifests describe the selected inputs; they do not grant rights to those inputs.

The EfficientNet implementation is imported from [torchvision](https://github.com/pytorch/vision), and training can request its ImageNet-pretrained weights. Neither model implementation source nor pretrained weights are vendored here. Users should review the applicable software and weight/dataset terms before redistribution. The architecture is attributed to Tan and Le (2019), linked in the README.

NumPy, SciPy, pandas, Pillow, scikit-learn, PyTorch, torchvision, Matplotlib, and Jupyter are external dependencies with their own licenses. This release includes project code that calls their APIs rather than copies of their implementations. The figures are generated from the project's saved confusion matrices; standard Matplotlib plotting and font rendering do not make the underlying scientific results third-party predictions.

Selected historical text logs retain the project's saved numerical outputs. Full historical working notebooks and their environment output are not distributed. External publications, including Smith and Randall (2015), are cited for their findings rather than included as licensed project content. No vendored third-party implementation or downloaded external image asset was identified in the release review.
