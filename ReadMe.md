<img src="docs/source/images/logo.png" width="15%" align="left">

# Towards foundational LiDAR world models with efficient latent flow matching

<p align="center">
  <strong>NeurIPS 2025</strong>
</p>

<p align="center">
  <a target="_blank">Tianran Liu</a>&nbsp;&nbsp;&nbsp;
  <a target="_blank">Shengwen Zhao</a>&nbsp;&nbsp;&nbsp;
  <a href="https://leaf.utias.utoronto.ca/author/nicholas-rhinehart/" target="_blank">Nicholas Rhinehart</a>&nbsp;&nbsp;&nbsp;
    
  <br />
  Robotics Institute, University of Toronto&nbsp;&nbsp;&nbsp;
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.23434"><img src="https://img.shields.io/badge/Paper-PDF-B82A24?style=for-the-badge&logo=adobe-acrobat-reader" alt="Paper"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://orbis36.github.io/OccFM-NeurIPS2025/"><img src="https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=your-project-logo" alt="Project"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/Orbis36/OccFM-NeurIPS2025"> <img src="https://komarev.com/ghpvc/?username=Orbis36&repo=OccFM-NeurIPS2025&color=blue&style=for-the-badge" alt="visitors"> </a>
</p>

## TODO List
- [x] Release codes
- [x] Semantic forecasting weights release
- [x] Weights for occupancy FVD measurement

## Setup environment
```shell
conda create -n OccFM python=3.10 && conda activate OccFM
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install nuscenes_devkit matplotlib==3.10.3 einops einops_exts pyyaml easydict wandb rich
```
## Download cached latent & pickle files
Please download the cached latent from [Hugging Face](https://huggingface.co/Orbis36/OccFM/tree/main).

The data and logs folder should looks like: 
```text
OccFM/data/nusc_latent_vae/
├── x16
│   ├── nus_sem_occ_training.pkl
│   └── nus_sem_occ_validation.pkl
```

```text
OccFM/logs/
├── occfm/
│   └── 2s_3s_nusc_hist_traj/
│       ├── ckpt/
│       │   ├── epoch=000137.ckpt
│       │   ├── epoch=000171.ckpt
│       │   └── epoch=000199.ckpt
│       └── occfm.yaml
│
├── occfm_3dvae/ * for FID/FVD eval
│   └── ori/
│       ├── ckpt/
│       │   ├── epoch=000038.ckpt
│       │   ├── epoch=000039.ckpt
│       │   └── epoch=000040.ckpt
│       └── occfm_3dvae.yaml
│
├── occfm_vae/ * Proposed vae in Figure 2
│   └── 100ep_3docc_sem_voxel/
│       ├── ckpt/
│       │   ├── epoch=000100.ckpt
│       └── occfm_vae.yaml
│
└── occfm_fut/
    └── 2s_3s_nusc_fut_traj/
        ├── ckpt/
        │   ├── epoch=000098.ckpt
        │   ├── epoch=000107.ckpt
        │   └── epoch=000196.ckpt
        └── occfm_fut.yaml
```

## Model Training / Evaluation

### VAE for data compression

Eval compress results:
```shell
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master-port=19502 tools/train.py \ 
  --cfg_file tools/cfgs/occfm_vae.yaml \
  --extra_tag 100ep_3docc_sem_voxel \
  --fix_random_seed \
  --amp \
  --ckpt ./logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --skip_opti \
  --eval_mode
```

### 4D Semantic Occupancy Forecasting 

Train model from scratch:
```shell
python -m torch.distributed.run \
  --nproc_per_node=<num of GPU> \
  --master-port=19502 \
  tools/train.py \
  --cfg_file tools/cfgs/occfm_fut.yaml \
  --extra_tag 2s_3s_nusc_fut_traj \
  --fix_random_seed \
  --use_ema \
  --amp
```

Eval forecasting results with future trajectory, 2s to 3s forecasting:
```shell
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master-port=19502 \
  tools/train.py \
  --cfg_file tools/cfgs/occfm_fut.yaml \
  --extra_tag 2s_3s_nusc_fut_traj \
  --fix_random_seed \
  --use_ema \
  --amp \
  --ckpt ./logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --eval_mode
```

Eval forecasting results without future trajectory, 2s to 3s forecasting:

```shell
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master-port=19502 \
  tools/train.py \
  --cfg_file tools/cfgs/occfm.yaml \
  --extra_tag 2s_3s_nusc_hist_traj \
  --fix_random_seed \
  --use_ema \
  --amp \
  --ckpt ./logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --eval_mode
```

### Eval the temporal consistency of your Occ-based World Model

Benefiting from the low informantion loss during VAE reconstruction that we propose, it is in fact highly suitable for evaluating the generative quality of OCC models using the Inception method.
Specifically, we incorporated temporal processing components into both the encoder and decoder stages to enable the network to capture sequential information for calculating the FVD.

The time window for calculating the FVD weight in our release is 6. You must first cache all predicted occupancy data. 

- For nuscenes, the size is [6, 200, 200, 16], with each batch stored as a separate file. Finally, compute the FVD using the following command, replacing `path` with the actual directory where your files are stored.
- For FID, please use the weights and YAML file for the VAE under 100ep_3docc_sem_voxel.
```shell
python tools/test_fid.py
--eval_model
<path to the occupancy forecasted>
--fix_random_seed
--ckpt
./logs/occfm_3dvae/ori/ckpt/epoch=000040.ckpt
--cfg_file
./tools/cfgs/occfm_3dvae.yaml
```

### Citation
If you find our work interesting, don't forget to cite us:
```text
@article{liu2025towards,
  title={Towards foundational LiDAR world models with efficient latent flow matching},
  author={Liu, Tianran and Zhao, Shengwen and Rhinehart, Nicholas},
  journal={arXiv preprint arXiv:2506.23434},
  year={2025}
}
```




