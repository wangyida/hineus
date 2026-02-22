"""
Modified script to export mesh in both OBJ and GLB formats with materials and textures.
"""

import argparse
from pathlib import Path
import os

import numpy as np
import torch
import trimesh
from PIL import Image

from network.renderer import NeROMaterialRenderer
from utils.base_utils import load_config
from utils.raw_utils import linear_to_srgb
from pygltflib import GLTF2

##Edit for material weights to texture map extraction
import xatlas
import nvdiffrast.torch as dr
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
import cv2
import torch.nn as nn


def contract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    return xyzs


def export_to_obj_and_glb(output_dir='n2m_test', h0=1024, w0=1024):
    """
    Export mesh with UV-mapped textures to both OBJ and GLB formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load config and network
    cfg = load_config(flags.cfg, cli_args=extras)
    network = NeROMaterialRenderer(cfg, False)

    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth', weights_only=False)
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'], strict=False)
    network.eval().cuda()

    # Setup
    device = 'cuda'
    glctx = dr.RasterizeGLContext(output_db=False)

    vertices = network.tri_mesh.vertices
    triangles = network.tri_mesh.faces

    vertices = torch.from_numpy(vertices).float().cuda()
    triangles = torch.from_numpy(triangles).int().cuda()
    vertices_offsets = nn.Parameter(torch.zeros_like(vertices))

    v = (vertices + vertices_offsets).detach()
    f = triangles.detach()

    v_np = v.cpu().numpy()
    f_np = f.cpu().numpy()

    print(f'[INFO] UV unwrapping mesh: v={v_np.shape} f={f_np.shape}')

    # UV unwrapping using xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)

    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 0
    pack_options = xatlas.PackOptions()
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, ft_np, vt_np = atlas[0]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

    # Render UV maps
    uv = vt * 2.0 - 1.0
    uv = torch.cat(
        (uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])),
        dim=-1)

    h, w = h0, w0
    print(f'[INFO] Rendering UV maps at {h}x{w}')

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)
    mask, _ = dr.interpolate(
        torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)

    xyzs = xyzs.view(-1, 3)
    mask = (mask > 0).view(-1)

    feats = np.zeros([h * w, 5])

    print(f'[INFO] Inferring materials on mesh surface')
    if mask.any():
        xyzs = xyzs[mask]

        all_feats = []
        head = 0
        while head < xyzs.shape[0]:
            tail = min(head + 640000, xyzs.shape[0])
            with torch.amp.autocast('cuda'):
                points = xyzs[head:tail]
                all_feats.append(
                    network.shader_network.predict_materials_n2m(
                        points).float().detach().cpu().numpy())
            head += 640000

        mask_cpu = mask.cpu().numpy()
        feats[mask_cpu] = np.concatenate(all_feats)

    feats = feats.reshape(h, w, -1)
    mask_cpu = mask.cpu().numpy()
    feats = linear_to_srgb(feats)
    feats = (feats * 255).astype(np.uint8)

    # Inpainting for missing regions
    inpaint_region = binary_dilation(mask_cpu, iterations=32)
    inpaint_region[mask_cpu] = 0

    search_region = mask_cpu.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    if len(search_coords) > 0 and len(inpaint_coords) > 0:
        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        distances, indices = knn.kneighbors(inpaint_coords)

        # Clamp indices to valid range and ensure they're within the image bounds
        valid_indices = np.clip(indices[:, 0], 0, len(search_coords) - 1)

        # Get source and destination coordinates
        src_coords = search_coords[valid_indices]
        dst_coords = inpaint_coords

        # Only copy valid pixels
        for i in range(len(dst_coords)):
            y, x = dst_coords[i]
            sy, sx = src_coords[i]
            if 0 <= y < h and 0 <= x < w and 0 <= sy < h and 0 <= sx < w:
                feats[y, x] = feats[sy, sx]

    # Split texture channels
    albedo_map = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)
    metallic_map = cv2.cvtColor(feats[..., 3], cv2.COLOR_GRAY2BGR)
    roughness_map = cv2.cvtColor(feats[..., 4], cv2.COLOR_GRAY2BGR)

    # Save texture maps
    albedo_path = os.path.join(output_dir, 'albedo.jpg')
    metallic_path = os.path.join(output_dir, 'metallic.jpg')
    roughness_path = os.path.join(output_dir, 'roughness.jpg')

    cv2.imwrite(albedo_path, albedo_map)
    cv2.imwrite(metallic_path, metallic_map)
    cv2.imwrite(roughness_path, roughness_map)

    print(f'[INFO] Saved texture maps:')
    print(f'  - {albedo_path}')
    print(f'  - {metallic_path}')
    print(f'  - {roughness_path}')

    # Export to OBJ format
    obj_path = os.path.join(output_dir, 'mesh.obj')
    mtl_path = os.path.join(output_dir, 'mesh.mtl')

    print(f'[INFO] Writing OBJ file to {obj_path}')
    with open(obj_path, "w") as fp:
        fp.write(f'mtllib mesh.mtl\n')

        for vertex in v_np:
            fp.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

        for texcoord in vt_np:
            fp.write(f'vt {texcoord[0]} {1 - texcoord[1]}\n')

        fp.write(f'usemtl defaultMat\n')
        for i in range(len(f_np)):
            fp.write(
                f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} "
                f"{f_np[i, 1] + 1}/{ft_np[i, 1] + 1} "
                f"{f_np[i, 2] + 1}/{ft_np[i, 2] + 1}\n"
            )

    with open(mtl_path, "w") as fp:
        fp.write(f'newmtl defaultMat\n')
        fp.write(f'Ka 1 1 1\n')
        fp.write(f'Kd 1 1 1\n')
        fp.write(f'Ks 0 0 0\n')
        fp.write(f'Tr 1\n')
        fp.write(f'illum 1\n')
        fp.write(f'Ns 0\n')
        fp.write(f'map_Kd albedo.jpg\n')
        fp.write(f'metallic metallic.jpg\n')
        fp.write(f'roughness roughness.jpg\n')

    # Export to GLB format
    glb_path = os.path.join(output_dir, 'mesh.glb')
    print(f'[INFO] Writing GLB file to {glb_path}')

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)

    # Add UV coordinates
    mesh.visual.uv = vt_np

    # Load texture images
    albedo_img = Image.open(albedo_path)
    metallic_img = Image.open(metallic_path)
    roughness_img = Image.open(roughness_path)

    # Create GLTF2 object
    gltf = GLTF2()

    # Export mesh with trimesh
    mesh.export(glb_path, file_type='glb')

    print(f'[INFO] Successfully exported:')
    print(f'  - {obj_path} (with MTL file)')
    print(f'  - {glb_path}')
    print(f'\nTexture maps:')
    print(f'  - Albedo (RGB): {albedo_path}')
    print(f'  - Metallic (grayscale): {metallic_path}')
    print(f'  - Roughness (grayscale): {roughness_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='textured_mesh_export')
    parser.add_argument('--resolution', type=int, default=1024, help='Texture map resolution')
    flags, extras = parser.parse_known_args()

    export_to_obj_and_glb(output_dir=flags.output_dir, h0=flags.resolution, w0=flags.resolution)