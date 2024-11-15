# Grap2Eq

This repository contains the PyTorch implementation for non-diffusion-based Grap2Eq.

## Dependencies
Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard
* CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our method is quantitatively evaluated on [Human3.6M](http://vision.imar.ro/human3.6m), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) and [HumanEva](http://humaneva.is.tue.mpg.de/) datasets. 

### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints detected by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO). You can download our processed data from [here]([https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link](https://drive.google.com/file/d/11eBe175Rgj6IYrwZwa1oXTOyHPxGuWyi/view)). Put them in the `./data` directory. 

### HumanEva-I
We set up the HumanEva-I dataset similar to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). You can download the 2D ground truths from [here](https://drive.google.com/file/d/1UuW6iTdceNvhjEY2rFF9mzW93Fi1gMtz/view), and the 3D ground truths from [here](https://drive.google.com/file/d/1CtAJR_wTwfh4rEjQKKmABunkyQrvZ6tu/view). `data_2d_humaneva15_gt.npz` is the ground truth of 2D keypoints. `data_3d_humaneva15.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

## Training and Evaluation

We trained our models on 1*NVIDIA RTX 4090.

To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python run.py -k cpn_ft_h36m_dbb -f 243 -s 243 -l log/run -c checkpoint -gpu 0
```

To evaluate our Grap2Eq using the 2D keypoints obtained by CPN as inputs, please run:
```bashpython run.py -k cpn_ft_h36m_dbb -c <checkpoint_path> --evaluate <checkpoint_file> -f 243 -s 243
```
## Acknowledgement
Our code refers to the following repositories.
* [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse)
* [D3DP](https://github.com/paTRICK-swk/D3DP)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [FinePOSE](https://github.com/PKU-ICST-MIPL/FinePOSE_CVPR2024)
* [KTPFormer](https://github.com/JihuaPeng/KTPFormer)
* [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
* [MotionBERT](https://github.com/Walter0807/MotionBERT)

We thank the authors for releasing their codes.


