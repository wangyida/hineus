import numpy as np
import torch
from network.field import SDFNetwork, NeRFNetwork

class PlanarLoss(nn.Module):
    """
    Local Geometry-Constrained Regularization
    """
    default_cfg = {
        "K": 8,             # Number of neighbors
        'eta': 0.05,        # Sampling radius
        'epsilon': 1e-3,    # Numerical stability
    }

    def __init__(self, cfg, sdf_network: SDFNetwork, color_network: NeRFNetwork):
        """
        Args:
            cfg: Configuration dictionary.
            sdf_network: The active SDFNetwork instance.
            color_network: The active NeRFNetwork instance.
        """
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        
        # Store references to the active network instances
        self.sdf_network = sdf_network
        self.color_network = color_network

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        """
        Args:
            data_pr: Dictionary containing predicted data. 
                     Requires 'inter_sdf', 'rays_o', 'rays_d'.
            data_gt: Dictionary containing ground truth.
            step: Current training step.
        """
        # Extract data
        # 'inter_sdf' presents the surface intersection point x0
        x0 = data_pr['inter_sdf']   # [B, 3]
        ray_o = data_pr['rays_o']   # [B, 3]
        ray_v = data_pr['rays_d']   # [B, 3]

        # Derive depth (t0) and normal (n0) from x0
        # We calculate these on-the-fly to ensure they correspond exactly to x0 (inter_sdf)
        # rather than using volume-rendered approximations that might exist in data_pr.
        
        # t0 = projection of (x0 - ray_o) onto ray_v
        # Assuming ray_v is normalized
        t0 = torch.sum((x0 - ray_o) * ray_v, dim=-1, keepdim=True) # [B, 1]

        # n0 = Gradient of SDF at x0
        # We compute this explicitly to ensure we have the normal at the zero-crossing
        n0 = self.sdf_network.gradient(x0)
        n0 = F.normalize(n0, dim=-1) # [B, 3]

        # Hyperparameters
        K = self.cfg['K']
        eta = self.cfg['eta']
        epsilon = self.cfg['epsilon']

        batch_size = x0.shape[0]
        device = x0.device

        # Sample neighboring points, delta_t ~ uniform(-eta, 0)
        delta_t = torch.rand(batch_size, K, device=device) * (-eta)
        
        # t_k = t0 + delta_t
        t_k = t0 + delta_t # [B, K]
        
        # x_k = ray_o + t_k * ray_v
        # [B, K, 3]
        x_k = ray_o.unsqueeze(1) + t_k.unsqueeze(-1) * ray_v.unsqueeze(1)
        x_k_flat = x_k.reshape(-1, 3)
        
        # Compute geometric planarity term
        sdf_k, _ = self.sdf_network(x_k_flat)
        
        f_xk = sdf_k.reshape(batch_size, K) # [B, K]
        
        # Prepare broadcasting for x0 and n0
        x0_expanded = x0.unsqueeze(1).expand(-1, K, -1) # [B, K, 3]
        n0_expanded = n0.unsqueeze(1).expand(-1, K, -1) # [B, K, 3]
        
        # Distance ||x_k - x0||
        diff_vec = x_k - x0_expanded
        dist_k = torch.norm(diff_vec, dim=-1) + 1e-8
        
        # Planarity error: | f(x_k)/dist - n0 . (x_k - x0)/dist |
        term_a = f_xk / dist_k
        term_b = torch.sum(n0_expanded * (diff_vec / dist_k.unsqueeze(-1)), dim=-1)
        planar_error = torch.abs(term_a - term_b) # [B, K]

        # Compute adaptive weighting: get features for neighbors x_k
        feat_xk = self._get_radiance_features(x_k_flat) # [B*K, C]
        feat_xk = feat_xk.reshape(batch_size, K, -1)
        
        # Get features for surface point x0
        feat_x0 = self._get_radiance_features(x0) # [B, C]
        feat_x0_expanded = feat_x0.unsqueeze(1).expand(-1, K, -1)
        
        # Feature distance
        feat_diff = torch.norm(feat_xk - feat_x0_expanded, dim=-1) # [B, K]
        
        # Calculate lambda
        # Detach to prevent optimizing features to minimize this weight
        lambda_pla_k = epsilon / (feat_diff + epsilon)
        lambda_pla_k = lambda_pla_k.detach()
        
        # Loss aggregation
        loss_planar = torch.mean(lambda_pla_k * planar_error)
        
        # Average lambda per ray for Eikonal usage
        lambda_pla_per_ray = torch.mean(lambda_pla_k, dim=1, keepdim=True)
        
        return loss_planar, lambda_pla_per_ray

    def _get_radiance_features(self, x):
        """
        Computes radiance features for points x.
        """
        # Enable gradients for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            
            # Get SDF and geometry features: SDFNetwork.forward(x) -> (sdf, feat)
            sdf, geo_feat = self.sdf_network(x)

            # Compute gradients (normals)
            gradients = self.sdf_network.gradient(x)
            normals = F.normalize(gradients, dim=-1)

        # View direction assuming v_k follows the direction of the sdf gradient (normal)
        view_dirs = normals

        # Query color
        features = self.color_network(x, normals, view_dirs, geo_feat)
        
        return features


class Loss:
    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class NeRFRenderLoss(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_rgb' in data_pr: outputs['loss_rgb'] = data_pr['loss_rgb']
        if 'loss_rgb_fine' in data_pr: outputs['loss_rgb_fine'] = data_pr['loss_rgb_fine']
        if 'loss_global_rgb' in data_pr: outputs['loss_global_rgb'] = data_pr['loss_global_rgb']
        if 'loss_rgb_inner' in data_pr: outputs['loss_rgb_inner'] = data_pr['loss_rgb_inner']
        if 'loss_rgb0' in data_pr: outputs['loss_rgb0'] = data_pr['loss_rgb0']
        if 'loss_rgb1' in data_pr: outputs['loss_rgb1'] = data_pr['loss_rgb1']
        if 'loss_masks' in data_pr: outputs['loss_masks'] = data_pr['loss_masks']
        return outputs


class EikonalLoss(Loss):
    default_cfg = {
        "eikonal_weight": 0.1,
        'eikonal_weight_anneal_begin': 0,
        'eikonal_weight_anneal_end': 0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def get_eikonal_weight(self, step):
        if step < self.cfg['eikonal_weight_anneal_begin']:
            return 0.0
        elif self.cfg['eikonal_weight_anneal_begin'] <= step < self.cfg['eikonal_weight_anneal_end']:
            return self.cfg['eikonal_weight'] * (step - self.cfg['eikonal_weight_anneal_begin']) / \
                (self.cfg['eikonal_weight_anneal_end'] - self.cfg['eikonal_weight_anneal_begin'])
        else:
            return self.cfg['eikonal_weight']

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        weight = self.get_eikonal_weight(step)
        method = 'neus'
        if method == 'raneus' and 'inner_mask' in data_pr:
            pts_per_ray = torch.sum(torch.tensor(data_pr['inner_mask']).int(), -1)
            factor_r = 0.001
            with torch.no_grad():
                diff_rays = factor_r / (data_pr['loss_rgb'] + factor_r)
            diff_pts = torch.repeat_interleave(diff_rays, pts_per_ray)
            if data_pr['gradient_error'].size()[0] == diff_pts.size()[0]:
                data_pr['gradient_error'] *= diff_pts
        outputs = {'loss_eikonal': data_pr['gradient_error'] * weight}
        return outputs


class MaterialRegLoss(Loss):
    default_cfg = {
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_mat_reg' in data_pr: outputs['loss_mat_reg'] = data_pr['loss_mat_reg']
        if 'loss_diffuse_light' in data_pr: outputs['loss_diffuse_light'] = data_pr['loss_diffuse_light']
        return outputs


class StdRecorder(Loss):
    default_cfg = {
        'apply_std_loss': False,
        'std_loss_weight': 0.05,
        'std_loss_weight_type': 'constant',
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'std' in data_pr:
            outputs['std'] = data_pr['std']
            if self.cfg['apply_std_loss']:
                if self.cfg['std_loss_weight_type'] == 'constant':
                    outputs['loss_std'] = data_pr['std'] * self.cfg['std_loss_weight']
                else:
                    raise NotImplementedError
        if 'inner_std' in data_pr: outputs['inner_std'] = data_pr['inner_std']
        if 'outer_std' in data_pr: outputs['outer_std'] = data_pr['outer_std']
        return outputs


class OccLoss(Loss):
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_occ' in data_pr:
            outputs['loss_occ'] = torch.mean(data_pr['loss_occ']).reshape(1)
        return outputs


class InitSDFRegLoss(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        reg_step = 1000
        small_threshold = 0.1
        large_threshold = 1.05
        if 'sdf_vals' in data_pr and 'sdf_pts' in data_pr and step < reg_step:
            norm = torch.norm(data_pr['sdf_pts'], dim=-1)
            sdf = data_pr['sdf_vals']
            small_mask = norm < small_threshold
            if torch.sum(small_mask) > 0:
                bounds = norm[small_mask] - small_threshold  # 0-small_threshold -> 0
                # we want sdf - bounds < 0
                small_loss = torch.mean(torch.clamp(sdf[small_mask] - bounds, min=0.0))
                small_loss = torch.sum(small_loss) / (torch.sum(small_loss > 1e-5) + 1e-3)
            else:
                small_loss = torch.zeros(1)

            large_mask = norm > large_threshold
            if torch.sum(large_mask) > 0:
                bounds = norm[large_mask] - large_threshold  # 0 -> 1 - large_threshold
                # we want sdf - bounds > 0 => bounds - sdf < 0
                large_loss = torch.clamp(bounds - sdf[large_mask], min=0.0)
                large_loss = torch.sum(large_loss) / (torch.sum(large_loss > 1e-5) + 1e-3)
            else:
                large_loss = torch.zeros(1)

            anneal_weights = (np.cos((step / reg_step) * np.pi) + 1) / 2
            return {'loss_sdf_large': large_loss * anneal_weights, 'loss_sdf_small': small_loss * anneal_weights}
        else:
            return {}

class MaskLoss(Loss):
    default_cfg = {
        'mask_loss_weight': 0.01,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_mask' in data_pr:
            outputs['loss_mask'] = data_pr['loss_mask'].reshape(1) * self.cfg['mask_loss_weight']
        return outputs


name2loss = {
    'nerf_render': NeRFRenderLoss,
    'eikonal': EikonalLoss,
    'std': StdRecorder,
    'init_sdf_reg': InitSDFRegLoss,
    'occ': OccLoss,
    'mask': MaskLoss,

    'mat_reg': MaterialRegLoss,
}
