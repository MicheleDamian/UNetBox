# UNetBox

---

![Network Schema](./schema.png)

**UNetBox** is a PyTorch neural network for image segmentation that improves the popular UNet framework. It provides a box of techniques that are commonly used in the computer-vision field, but weren't included in UNet, in the form of plugins that can be easily enabled/disable. As for now, the following are implemented inside UNetBox:

* Sigmoid linear unit (SiLU / Swish)
* BatchNorm
* Squeeze Excitation layer
* Convolution Transposed
* Dimensional Expansion and Compression after downsampling/upsampling

The goal of the project is to add more techniques, as they become available, in order to push the state-of-the-art in image segmentation.


## Ablation Study

In order to explore the influence of the components on UNetBox's performance, the following table shows a set of ablation studies on [Google's contrails identification dataset](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming).

![Contrails dataset example](./contrails_dataset_example.png)

All tests train with the Adam optimizer on batches of 64 samples, using mixed precision, and a 1-cycle cosine scheduler to set the learning rate. Training run until convergence of the *focal loss* on an out-of-sample dataset (3-fold cross validation).

| SI | SE | EC | BN | CT | Focal Loss x10<sup>-3</sup> | Standard Error | Training Time (m) | # Parameters |
|:--:|:--:|:--:|:--:|:--:|----------------------------:|---------------:|------------------:|-------------:|
| ✘  | ✘  | ✘  | ✘  | ✘  |                       1.065 |         ± .006 |               242 |            - |
| ✘  | ✘  | ✘  | ✘  | ✔  |                       1.081 |         ± .029 |               195 |    3,032,017 |
| ✘  | ✔  | ✘  | ✘  | ✘  |                        1.08 |         ± .082 |               226 |    3,087,317 |
| ✘  | ✘  | ✘  | ✔  | ✔  |                       1.057 |         ± .008 |               132 |    3,033,937 |
| ✘  | ✘  | ✔  | ✔  | ✔  |                        1.04 |         ± .005 |               163 |    6,063,761 |
| ✔  | ✘  | ✔  | ✔  | ✔  |                       1.022 |         ± .022 |               147 |    6,063,761 |
| ✔  | ✔  | ✔  | ✔  | ✔  |                       1.016 |         ± .014 |               132 |    6,119,061 |

Definitions:

* *SI* : SiLU activation units
* *SE* : Squeeze Excitation block added to last activation tensor before downsampling/upsampling
* *EC* : Expansion followed by a Compression of the channels added after downsampling/upsampling
* *BN* : BatchNorm
* *CT* : Convolution Transposed replaces bilinear interpolation for upsampling 

A Jupyter Notebook that runs all the tests in the table is provided at [tests/contrails_ablation_study.ipynb](./tests/contrails_ablation_study.ipynb). You can refer to it as an example to use the package as well. 
