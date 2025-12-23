import argparse

from pathlib import Path

import torch
import trimesh
from network.field import extract_geometry

from network.renderer import name2renderer
from utils.base_utils import load_cfg, load_config


def main():
    cfg = load_config(flags.cfg, cli_args=extras)
    network = name2renderer[cfg['network']](cfg, training=False)

    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'successfully load {cfg["name"]} step {step}!')

    bbox_min = -torch.ones(3)
    bbox_max = torch.ones(3)
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            bbox_min, bbox_max, flags.resolution, 0,
            lambda x: network.sdf_network.sdf(x))
    import numpy as np
    if cfg["database_type"] == 'syn' or cfg["database_type"] == 'nerf':
        object_name = cfg["database_name"].split('/')
        # output geometry
        vertices /= cfg["scale_factor"]
        mesh = trimesh.Trimesh(vertices,
                               triangles,
                               vertex_normals=np.ones_like(vertices))
        output_dir = Path('data/meshes')
        output_dir.mkdir(exist_ok=True)
        mesh.export(str(output_dir / f'{cfg["name"]}-{step}.ply'))
    else:
        object_name, max_len = cfg["database_name"].split('/')
        # output geometry
        mesh = trimesh.Trimesh(vertices,
                               triangles,
                               vertex_normals=np.ones_like(vertices))
        mesh.export(
            f'{cfg["dataset_dir"]}/{object_name}/sparse/0/points3D_repose.ply')
        repose = np.load(f'{cfg["dataset_dir"]}/{object_name}' +
                         '/repose.npy.npz')
        R = repose['arr_0']
        t = repose['arr_1']
        s = repose['arr_2']
        vertices /= s
        vertices = np.transpose(np.linalg.inv(R) @ np.transpose(vertices))
        vertices -= t

        # output geometry
        mesh = trimesh.Trimesh(vertices,
                               triangles,
                               vertex_normals=np.ones_like(vertices))
        mesh.visual.vertex_colors = [0, 255, 0, 50]

        mesh.export(
            f'{cfg["dataset_dir"]}/{object_name}/sparse/0/points3D_mesh.ply')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    flags, extras = parser.parse_known_args()
    main()
