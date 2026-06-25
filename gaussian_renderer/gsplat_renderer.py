import torch
import math
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene using gsplat.
    """
    if not GSPLAT_AVAILABLE:
        raise ImportError("gsplat is not installed")

    # extracting K matrix
    fx = (viewpoint_camera.image_width / 2.0) / math.tan(viewpoint_camera.FoVx / 2.0)
    fy = (viewpoint_camera.image_height / 2.0) / math.tan(viewpoint_camera.FoVy / 2.0)
    K = torch.tensor([[fx, 0, viewpoint_camera.image_width / 2.0],
                      [0, fy, viewpoint_camera.image_height / 2.0],
                      [0,  0, 1.0]], device="cuda", dtype=torch.float32).unsqueeze(0)
    
    # extracting view matrix
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)
    
    bg = bg_color.unsqueeze(0)
    
    # Calling gsplat
    shs = pc.get_features if override_color is None else override_color
    
    out_colors, alphas, meta = rasterization(
        means=pc.get_xyz,
        quats=pc.get_rotation,
        scales=pc.get_scaling * scaling_modifier,
        opacities=pc.get_opacity.squeeze(-1) if pc.get_opacity.dim() == 2 else pc.get_opacity,
        colors=shs,
        viewmats=viewmat,
        Ks=K,
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        sh_degree=pc.active_sh_degree,
        render_mode="RGB+ED",
        backgrounds=bg
    )
    
    # image and depth
    image = out_colors[0, :, :, :3].permute(2, 0, 1) 
    depth_image = out_colors[0, :, :, 3].unsqueeze(0)
    
    # meta info
    radii = meta["radii"].squeeze(0)
    viewspace_points = meta["means2d"].squeeze(0)
    visibility_filter = radii > 0
    
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        image = torch.matmul(image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]
    
    image = image.clamp(0, 1)

    out = {
        "render": image,
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "true_radii": radii.detach().cpu(), # gsplat currently doesn't output true_radii
        "depth": depth_image,
        "is_contributing": visibility_filter.detach().cpu(),
        "contributions": None  # No direct mapping available, use dummy values
    }
    
    return out
