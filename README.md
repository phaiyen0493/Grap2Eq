# Diffusion-based Grap2Eq

This repository contains the PyTorch implementation for Grap2Eq.

![Grap2Eq](https://github.com/user-attachments/assets/04654725-04e8-4d2c-95a9-2c34f654b074)

## Comparison with SOTA methods on In-the-Wild videos

Check out our video:

https://github.com/user-attachments/assets/b0472c93-661e-46ab-92a5-4c04bc4213bb

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
We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO) and [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main). You can download our processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 

### HumanEva-I
We set up the HumanEva-I dataset similar to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). You can download the 2D ground truths from [here](https://drive.google.com/file/d/1UuW6iTdceNvhjEY2rFF9mzW93Fi1gMtz/view), and the 3D ground truths from [here](https://drive.google.com/file/d/1CtAJR_wTwfh4rEjQKKmABunkyQrvZ6tu/view). `data_2d_humaneva15_gt.npz` is the ground truth of 2D keypoints. `data_3d_humaneva15.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

## Training and Evaluation

We trained our models on 1*NVIDIA RTX 4090.

### Human3.6M

#### 2D CPN inputs
To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main_h36m.py -k cpn_ft_h36m_dbb -c checkpoint/h36m -gpu 0 --nolog
```

To evaluate our Grap2Eq using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main_h36m.py -k cpn_ft_h36m_dbb -c checkpoint/h36m -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10 --p2
```

#### 2D ground truth inputs
To train our model using the 2D ground truth keypoints as inputs, please run:
```bash
python main_h36m.py -k gt -c checkpoint/h36m_gt -gpu 0 --nolog --save_lmin 21 --save_lmax 23
```

To evaluate our Grap2Eq using the 2D ground truth keypoints as inputs, please run:
```bash
python main_h36m.py -k gt -c checkpoint/h36m_gt -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10 --p2
```

### MPI-INF-3DHP
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/3dhp -gpu 0 --nolog
```

To evaluate our Grap2Eq using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/3dhp -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10
```
After that, the predicted 3D poses under P-Best, P-Agg, J-Best, J-Agg settings are saved as four files (`.mat`) in `./checkpoint`. To get the MPJPE, AUC, PCK metrics, you can evaluate the predictions by running a Matlab script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m` (you can change 'aggregation_mode' in line 29 to get results under different settings). Then, the evaluation results are saved in `./3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_{setting name}_t{iteration index}.csv`. You can manually average the three metrics in these files over six sequences to get the final results.

### HumanEva-I
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_humaneva.py -k gt -c 'checkpoint/humaneva_gt' -a 'Walk,Jog' -gpu 0 --nolog
```

To evaluate our Grap2Eq using the ground truth 2D poses as inputs, please run:
```bash
python main_humaneva.py -k gt -c 'checkpoint/humaneva_gt' -a 'Walk,Jog' -gpu 0 --nolog --evaluate <checkpoint_file> --by-subject -num_proposals 20 -sampling_timesteps 10 --p2
```

### Pretrained Models
[Google Drive](https://drive.google.com/drive/folders/1iEc6o7KlUfYpOYCN5Eo_rN0phLHnJrP1?usp=sharing)

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


