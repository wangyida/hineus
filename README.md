# HiNeuS
Relightable toolkit for reconstructing [3DRealCar](https://xiaobiaodu.github.io/3drealcar/) assets built on **High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity** 

![](images/drums.png)

[Project Page](https://wangyida.github.io/posts/hineus/) | [Paper](https://arxiv.org/abs/2506.23854) - ICCV 2025 **[Highlight]**

## Installation
```shell
git clone https://github.com/LiAutoAD/HiNeuS.git
cd HiNeuS
pip install -r requirements.txt
```
- Install `nvdiffrast` as instructed [https://nvlabs.github.io/nvdiffrast/#installation](https://nvlabs.github.io/nvdiffrast/#installation).
- Install `raytracing` as instructed [https://github.com/ashawkey/raytracing](https://github.com/ashawkey/raytracing).

## Geometric learning

### Training
General training API for nerf synthetic data and colmap project
```shell
# Nerf-synthetic data
python run_training.py --cfg configs/shape/nerf/general.yaml object=drums dataset_dir=${your-path}/dataset/nerf_synthetic

# COLMAP project
python run_training.py --cfg configs/shape/real/general.yaml object=sedan dataset_dir=${your-path}/dataset/real name={in-case-you-wanna-save-in-a-specifc-folder}
```
Intermediate results will be saved at `data/train_vis`. Models will be saved at `data/model`.

### Mesh extraction
```shell
python extract_mesh.py --cfg configs/shape/real/general.yaml object=sedan dataset_dir=${your-path}/dataset/real name={in-case-you-wanna-save-in-a-specifc-folder}
```
The extracted meshes will be saved at `data/meshes`.

## Material estimation

### Training
```shell
# estimate BRDF of the "sedan" of a general COLMAP project
python run_training.py --cfg configs/material/real/general.yaml object=sedan dataset_dir=${your-path}/dataset/real mesh=${your-mesh-path}.ply
```
Intermediate results will be saved at `data/train_vis`. Models will be saved at `data/model`.

### Extract materials
```shell
python extract_materials.py --cfg configs/material/syn/bell.yaml
python extract_materials.py --cfg configs/material/real/bear.yaml
```
The extracted materials will be saved at `data/materials`.

### Relighting
```shell
python relight.py --blender <path-to-your-blender> \
                  --name bell-neon \
                  --mesh data/meshes/bell_shape-300000.ply \
                  --material data/materials/bell_material-100000 \
                  --hdr data/hdr/neon_photostudio_4k.exr \
                  --trans
                  
python relight.py --blender <path-to-your-blender> \
                  --name bear-neon \
                  --mesh data/meshes/bear_shape-300000.ply \
                  --material data/materials/bear_material-100000 \
                  --hdr data/hdr/neon_photostudio_4k.exr
```
The relighting results will be saved at `data/relight` with the directory name of `bell-neon` or `bear-neon`. This command means that we use `neon_photostudio_4k.exr` to relight the object.

## Acknowledgement
Thanks for the open sourced project of [NeRO](https://liuyuan-pal.github.io/NeRO/) and [instant-angelo](https://github.com/hugoycj/Instant-angelo/) 
We highlight the practical adaptation for 3DGS assets in 3DRealCar dataset
![](images/autoassets_full.png)

## Citation
```
@inproceedings{wang2025hineus,
  title={HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity},
  author={Wang, Yida and Zhang, Xueyang and Zhan, Kun and Jia, Peng and Lang, Xianpeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}

@inproceedings{du20253drealcar,
  title={3drealcar: An in-the-wild rgb-d car dataset with 360-degree views},
  author={Du, Xiaobiao and Wang, Yida and Sun, Haiyang and Wu, Zhuojie and Sheng, Hongwei and Wang, Shuyun and Ying, Jiaying and Lu, Ming and Zhu, Tianqing and Zhan, Kun and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={26488--26498},
  year={2025}
}
```
