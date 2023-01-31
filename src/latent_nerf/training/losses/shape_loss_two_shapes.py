import torch
from torch import nn
from igl import read_obj

from src.latent_nerf.configs.train_config import GuideConfig
from src.latent_nerf.models.mesh_utils import MeshOBJ

DELTA = 0.2


def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)

    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()


class ShapeLoss(nn.Module):
    def __init__(self, cfg: GuideConfig):
        super().__init__()
        self.cfg = cfg
        v, _, _, f, _, _ = read_obj(self.cfg.shape_path1, float)
        mesh = MeshOBJ(v, f)
        v2, _, _, f2, _, _ = read_obj(self.cfg.shape_path2, float)
        mesh2 = MeshOBJ(v2, f2)
        self.sketchshape0 = mesh.normalize_mesh(cfg.mesh_scale)
        self.sketchshape = mesh2.normalize_mesh(cfg.mesh_scale)
        self.w = 0.55

    def forward(self, xyzs, sigmas):
        # Min of gaussian weighted distances
        mesh_occ0 = self.sketchshape0.winding_number(xyzs)
        mesh_occ1 = self.sketchshape.winding_number(xyzs)
        if self.cfg.proximal_surface > 0:
            weight0 = 1 - self.sketchshape0.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            weight1 = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            # weight = weight0 * self.w + weight1 * (1 - self.w)
            weight = torch.min(weight0, weight1)
        else:
            weight = None

        indicator0 = (mesh_occ0 > 0.5).float()
        indicator1 = (mesh_occ1 > 0.5).float()

        nerf_occ = 1 - torch.exp(-DELTA * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)

        # loss = ce_pq_loss(nerf_occ, indicator0, weight=weight)  # order is important for CE loss + second argument may not be optimized
        loss0 = ce_pq_loss(nerf_occ, indicator0,
                           weight=weight)  # order is important for CE loss + second argument may not be optimized
        loss1 = ce_pq_loss(nerf_occ, indicator1,
                           weight=weight)  # order is important for CE loss + second argument may not be optimized

        return loss0 * self.w + loss1 * (1 - self.w)

    def forward_copy(self, xyzs, sigmas):
        mesh_occ0 = self.sketchshape0.winding_number(xyzs)
        mesh_occ1 = self.sketchshape.winding_number(xyzs)
        if self.cfg.proximal_surface > 0:
            weight0 = 1 - self.sketchshape0.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            weight1 = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.cfg.proximal_surface)
            # weight = weight0 * self.w + weight1 * (1 - self.w)
            # weight = torch.min(weight0, weight1)
        else:
            weight = None
            weight0 = None
            weight1 = None

        indicator0 = (mesh_occ0 > 0.5).float()
        indicator1 = (mesh_occ1 > 0.5).float()

        nerf_occ = 1 - torch.exp(-DELTA * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)

        # loss = ce_pq_loss(nerf_occ, indicator0, weight=weight)  # order is important for CE loss + second argument may not be optimized
        loss0 = ce_pq_loss(nerf_occ, indicator0,
                           weight=weight0)  # order is important for CE loss + second argument may not be optimized
        loss1 = ce_pq_loss(nerf_occ, indicator1,
                           weight=weight1)  # order is important for CE loss + second argument may not be optimized

        return loss0 * self.w + loss1 * (1 - self.w)


    def apply_visual_hull(self, depths, views):
        self.sketchshape = self.sketchshape0.apply_visual_hull(depths, views, self.cfg.mesh_scale)

    def test_visual_hull(self, depths, views):
        return self.sketchshape0.test_visual_hull(depths, views, self.cfg.mesh_scale)
