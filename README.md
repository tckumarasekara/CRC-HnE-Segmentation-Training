# histologySegmentationTraining

![CPU Training Status](https://github.com/asd/seg_training/workflows/Train%20seg_training%20using%20CPU/badge.svg)
![Publish Container](https://github.com/asd/seg_training/workflows/Publish%20Container%20to%20Docker%20Packages/badge.svg)
![mlf-core lint](https://github.com/asd/seg_training/workflows/mlf-core%20lint/badge.svg)
![Documentation Status](https://readthedocs.org/projects/seg_training/badge/?version=latest)

Deep Learning training module for **semantic segmentation in histological images**.

The training dataset used is the **Lizard dataset**:
https://zenodo.org/record/7508237

The dataset comprises **4,981 patched images** from multiple colon tissue H&E-stained histological images.
Each image contains a segmentation mask with six nuclei classes:

- Neutrophil
- Epithelial
- Lymphocyte
- Plasma
- Eosinophil
- Connective tissue

Training can be performed both **deterministically** and **non-deterministically** on three different architectures:

- Basic U-Net
- Context U-Net
- Spatial Transformer U-Net

![Prediction Example](_images/pred_comb.png)

---

## License

- Free software: **MIT License**
- Documentation: https://seg-training.readthedocs.io

---

## Features

- Fully reproducible **mlf-core PyTorch model**
- Supports training with the following architectures:
  - U-Net
  - Context U-Net
  - Spatial Transformer U-Net

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd seg_training
```

2. Create the conda environment using the provided `environment_new.yml` file:

```bash
conda env create -f environment_new.yml
conda activate <environment-name>
```

## References

- **U-Net**: https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- **Context U-Net**: https://link.springer.com/chapter/10.1007/978-3-319-75238-9_25
- **mlf-core**: https://mlf-core.readthedocs.io/en/latest/
- **Cookiecutter**: https://github.com/audreyr/cookiecutter


---

## Credits

This package was created with **mlf-core** using **Cookiecutter**.

This repository was originally written by **Dominik Molitor**.
