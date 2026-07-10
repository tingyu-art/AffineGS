# AffineGS: Velocity Representation via Mixture of Affine Gaussians for Event-based Imaging Velocimetry
<!-- [![preprint](https://img.shields.io/static/v1?label=Journal&message=Submitted&color=B31B1B)](https://github.com/yongleex/INR-VG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->


This repository contains the official code for the submitted paper __AffineGS: Velocity Representation via Mixture of Affine Gaussians for Event-based Imaging Velocimetry__. This paper presents an efficient method for event-based imaging velocimetry based on affine Gaussian (AffineGS) to reconstruct velocity fields from sparse event streams.


## Methods
![Image](https://github.com/tingyu-art/AffineGS/blob/main/data/fig2.pdf)
The velocity field is modeled as a compact weighted superposition of localized anisotropic Gaussian primitives with continuous spatial support. Beyond a local velocity vector, each primitive is equipped with a velocity-gradient Jacobian matrix, which explicitly represents affine velocity variations within its neighborhood and improves the capacity to capture complex flow structures. The Gaussian parameters are estimated through test-time optimization in a contrast-maximization framework, where collapse-suppression regularization is incorporated to improve physical consistency. An adaptive density control strategy further refines the primitive distribution, while localized Gaussian evaluation and tile-based rendering reduce the computational cost.


## Install dependencies
``` bash
conda create -n affinegs python=3.10 -y
conda activate affinegs

# Install PyTorch. Please choose the CUDA version according to your local driver/toolkit.
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Common scientific and visualization packages.
conda install numpy scipy pandas matplotlib seaborn scikit-image h5py tqdm notebook ipykernel -c conda-forge
conda install opencv -c conda-forge
pip install addict
```

The CUDA extensions are optional but recommended for faster AffineGS optimization and rendering. If a compatible CUDA toolkit and compiler are available, they can be built as follows:

``` bash
cd methods/affinegs/accelerators/cmax_fused
python setup.py build_ext --inplace
cd ../../../../

cd methods/affinegs/accelerators/gaussian_renderer
python setup.py build_ext --inplace
cd ../../../../
```

If the extensions are not compiled successfully, the code will fall back to the PyTorch implementation, but the runtime may be slower.

## The experiments
* [EXP1.ipynb](https://github.com/tingyu-art/AffineGS/blob/main/experiments/EXP1.ipynb): evaluates the effectiveness of the affine Gaussian representation and its robustness to key event-generation factors, including particle density, noise level, and divergence regularization.
* [EXP2.ipynb](https://github.com/tingyu-art/AffineGS/blob/main/experiments/EXP2.ipynb): compares AffineGS with representative baseline methods on synthetic flow fields, including velocity accuracy and gradient-based quantities such as divergence and vorticity.
* [EXP3.ipynb](https://github.com/tingyu-art/AffineGS/blob/main/experiments/EXP3.ipynb): provides qualitative visual comparisons on real-world event data, highlighting the behavior of AffineGS and baseline methods in practical EBIV scenarios.


## Dataset

We evaluate the proposed method on both synthetic and real-world event-based datasets.

**Synthetic datasets.**  
Synthetic event streams are generated from predefined flow fields using an event simulation pipeline following our prior work in FED-PV [Wu et al., 2025]. The synthetic data include:
- simple analytical flow fields, and
- a large collection of flow fields derived from particle-based velocimetry datasets, adapted from the public PIV dataset repository:  
  [https://github.com/shengzesnail/PIV_dataset](https://github.com/shengzesnail/PIV_dataset
)

**Real-world datasets.**  
Real event data are captured using a practical event-based imaging velocimetry (EBIV) system. The evaluated flow scenarios include:
- a water-tank flow by our in-house EBIV setup, and
- a water-tank flow experiment by Willert and Klinner (2022).



## BibTeX
```
@article{li2026affinegs,
  title={AffineGS: Velocity Representation via Mixture of Affine Gaussians for Event-based Imaging Velocimetry},
  author={Li, Junjie and Lee, Yong},
  journal={Submitted},
  volume={},
  number={},
  year={2026},
  doi={},
  publisher={}
}
```

## Questions?
For any questions regarding this work, please email me at [aijia@whut.edu.cn](mailto:aijia@whut.edu.cn), or the correspondece [yonglee@whut.edu.cn](mailto:yonglee@whut.edu.cn).
