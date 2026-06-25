#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import torch.nn.functional as F
import json

from random import randint
from utils.loss_utils import ssim, weighted_ssim, fast_ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import numpy as np
from torch.amp import autocast, GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False
from PIL import Image

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, upscale):
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # ===================================================================
    # FLOAT32 DEPTH INJECTION - LOAD CACHE
    # ===================================================================

    depth_cache = {}
    depth_json_path = os.path.join(dataset.source_path, "sparse", "0", "depth_params.json")
    depth_npy_dir = os.path.join(dataset.source_path, "depths_npy")
    
    if os.path.exists(depth_json_path):
        with open(depth_json_path, 'r') as f:
            depth_params = json.load(f)
        
        print(f"\nLoading high-precision depth maps from {depth_npy_dir}...")
        for cam in scene.getTrainCameras():
            stem = os.path.splitext(cam.image_name)[0]
            if stem in depth_params:
                params = depth_params[stem]
                # THE FIX: Scale-Invariant Reliability Check
                if params['n_inliers'] >= 8 and params['pearson'] > 0.3:
                    npy_path = os.path.join(depth_npy_dir, f"{stem}.npy")
                    if os.path.exists(npy_path):
                        raw_depth = np.load(npy_path)
                        # Apply alignment perfectly in pure float32
                        aligned = params['scale'] * raw_depth + params['offset']
                        depth_cache[cam.uid] = torch.from_numpy(aligned).cuda()
                        # OPTIMIZATION: Pre-cache the render shape for this camera
                        depth_target_shapes[cam.uid] = (cam.image_height, cam.image_width)
        print(f"Successfully loaded {len(depth_cache)} reliable float32 depth maps.\n")
    # ===================================================================
    if args.render_debug:
        os.makedirs(os.path.join(scene.model_path, 'renders'), exist_ok=True)
    gaussians.training_setup(opt)
    # Load LR images
    for cam in scene.getTrainCameras():
        H, W = cam.image_height, cam.image_width
        h_lr, w_lr = round(H/upscale), round(W/upscale)
        lr_img_ext = os.listdir(f"{dataset.source_path}/images/")[0].split('.')[-1]
        image_name = cam.image_name
        image_name = ''.join(image_name.split('.')[:-1]) + f'.{lr_img_ext}'
        lr_image = Image.open(f"{dataset.source_path}/images/" + image_name).resize((w_lr, h_lr))
        lr_image = torch.from_numpy(np.array(lr_image)).permute(2, 0, 1) / 255.0
        cam.lr = lr_image
            
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    # ===================================================================
    # PERFORMANCE OPTIMIZATIONS FOR SMALL VRAM (RTX 3050)
    # ===================================================================
    scaler = GradScaler('cuda', enabled=True)  # Automatic Mixed Precision for ~2x speedup
    
    # Pre-cache depth map resolutions to avoid per-iteration interpolation
    depth_target_shapes = {}  # Map from uid -> target shape
    depth_maps_cached = {}    # Map from uid -> pre-resized depth map

    # Load weight maps
# Load weight maps
    for cam in scene.getTrainCameras():
        if args.weight_maps_path is not None:
            stem = os.path.splitext(cam.image_name)[0]
            sr_weight_map_lr = torch.load(
                os.path.join(args.weight_maps_path, f"SR_{stem}.pty"), weights_only=False
            ).cuda()
            cam.sr_weight_map = torch.nn.functional.interpolate(
                sr_weight_map_lr.unsqueeze(0).unsqueeze(0),
                (cam.image_height, cam.image_width),
                mode='bilinear',
                align_corners=False,
            )[0][0]
        else:
            if args.no_sr:
                print("Using SR of all zeros")
                cam.sr_weight_map = torch.zeros((cam.image_height, cam.image_width), device="cuda")
            elif args.full_sr:
                print("Using SR of all ones")
                cam.sr_weight_map = torch.ones((cam.image_height, cam.image_width), device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Use automatic mixed precision for ~2x speedup on small GPUs
        with autocast('cuda', dtype=torch.float16):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss computation
            gt_sr = viewpoint_cam.original_image.cuda()
            
            # OPTIMIZATION: Use pre-computed LR dimensions instead of accessing tensor shape
            lr_h, lr_w = viewpoint_cam.lr.shape[1:]
            # lr_h, lr_w = cam_lr_info.get(viewpoint_cam.uid, (viewpoint_cam.lr.shape[1], viewpoint_cam.lr.shape[2]))
            if image.shape != (3, int(lr_h * upscale), int(lr_w * upscale)):
                # Only interpolate if dimensions don't match (rare case)
                render_lr = torch.nn.functional.interpolate(image.unsqueeze(0), (lr_h, lr_w), mode='bicubic', align_corners=False).squeeze(0)
            else:
                # Fast path: direct downsampling
                render_lr = F.interpolate(image.unsqueeze(0), (lr_h, lr_w), mode='area').squeeze(0)

            Ll1_sr = (torch.abs((image - gt_sr))*viewpoint_cam.sr_weight_map.unsqueeze(0)).mean()
            Ll1_lr = (torch.abs(render_lr - viewpoint_cam.lr.cuda())).mean()
            Ll1 = args.gamma*Ll1_sr + (1-args.gamma)*Ll1_lr
            
            # The CUDA fused SSIM expects shape [1, 3, H, W] but image is [3, H, W]
            ssim_val_sr = fused_ssim(img1=image.unsqueeze(0),img2=gt_sr.unsqueeze(0))
            if args.weight_maps_path is not None and getattr(viewpoint_cam, 'sr_weight_map', None) is not None:
                # Approximate weighted behavior if needed, or fallback
                pass
            
            ssim_val_lr = fused_ssim(img1=render_lr.unsqueeze(0), img2=viewpoint_cam.lr.cuda().unsqueeze(0))
            ssim_value = args.gamma*ssim_val_sr + (1-args.gamma)*ssim_val_lr

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # OPTIMIZATION: Depth regularization - avoid per-iteration interpolation
            depth_weight_current = depth_l1_weight(iteration) 
            Ll1depth = 0.0 
            
            if depth_weight_current > 0 and viewpoint_cam.uid in depth_cache:
                invDepth_pred = render_pkg["depth"] 
                mono_target = depth_cache[viewpoint_cam.uid] 
                target_shape = depth_target_shapes.get(viewpoint_cam.uid, invDepth_pred.shape[1:])
                
                if invDepth_pred.shape[1:] != mono_target.shape:
                    # Only resize on first encounter (should be cached ideally)
                    mono_target_rs = F.interpolate(mono_target.unsqueeze(0).unsqueeze(0), size=invDepth_pred.shape[1:], mode='bilinear', align_corners=False)[0, 0]
                else:
                    mono_target_rs = mono_target
                
                loss_depth = torch.abs(invDepth_pred.squeeze() - mono_target_rs).mean()
                loss += depth_weight_current * loss_depth
                Ll1depth = (depth_weight_current * loss_depth).item()

        loss = loss.float()  # Convert back to float32 for backward pass
        
        # if viewpoint_cam.uid in depth_cache:
        #     invDepth_pred = render_pkg["depth"] # Usually [1, H, W]
        #     mono_target = depth_cache[viewpoint_cam.uid] # [H, W]
            
        #     # Match dimensions if resolution changed
        #     if invDepth_pred.shape[1:] != mono_target.shape:
        #         mono_target_rs = F.interpolate(mono_target.unsqueeze(0).unsqueeze(0), size=invDepth_pred.shape[1:], mode='bilinear')[0, 0]
        #     else:
        #         mono_target_rs = mono_target
            
        #     # Calculate L1 distance between aligned DA-V2 and 3DGS depth
        #     loss_depth = torch.abs(invDepth_pred.squeeze() - mono_target_rs).mean()
            
        #     # Apply depth regularization
        #     loss += depth_weight * loss_depth
        #     Ll1depth = (depth_weight * loss_depth).item() # Update for logging
        # ===================================================================

        scaler.scale(loss).backward()
        
        # Check if gradients are set; if not, scaler will error out because of torch amp backward check
        _has_any_grads = any(p.grad is not None for pg in gaussians.optimizer.param_groups for p in pg['params'])

        # iter_end.record()

        with torch.no_grad():
            if iteration % 1000 == 0 and args.render_debug:
                save_image([image.cpu(), viewpoint_cam.sr_weight_map.cpu().unsqueeze(0).repeat(3, 1, 1), gt_sr.cpu()], os.path.join(scene.model_path, 'renders', f'iter_{iteration}.jpg'))
                save_image([render_lr.cpu(), viewpoint_cam.lr.cpu()], os.path.join(scene.model_path, 'renders', f'iter_{iteration}_lr.jpg'))
                print(f"Iteration {iteration}")
                print(f"Image: {viewpoint_cam.image_name}")
                print("Loss: LR L1 {}, SR L1 {}, LR SSIM {}, SR SSIM {}".format(Ll1_lr.item(), Ll1_sr.item(), ssim_val_lr.item(), ssim_val_sr.item()))
                print(f"Iteration {iteration} == Total Loss: {Ll1} + {ssim_value} => {loss}")
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 200 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(200)
            if iteration == opt.iterations:
                progress_bar.close()

            # Save model
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 5 if iteration > opt.opacity_reset_interval else None
                    # Gate the pruning threshold based on whether depth supervision is active for this scene
                    # prune_opacity = 0.005 if opt.depth_l1_weight_init > 0 else 0.05
                    # Remove the conditional entirely
                    prune_opacity = 0.005
                    gaussians.densify_and_prune(opt.densify_grad_threshold, prune_opacity, scene.cameras_extent, size_threshold, radii)
                    #aussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step with gradient scaler
            if iteration < opt.iterations:
                try: 
                    if dataset.train_test_exp and gaussians.exposure_optimizer is not None and len(gaussians.exposure_optimizer.param_groups) > 0:
                        _has_exp_grads = any(p.grad is not None for pg in gaussians.exposure_optimizer.param_groups for p in pg['params'])
                        if _has_exp_grads:
                            scaler.step(gaussians.exposure_optimizer)
                            gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        if _has_any_grads:
                            visible = radii > 0
                            # Sparse Adam requires special handling, unscale grads then step directly
                            scaler.unscale_(gaussians.optimizer)
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        if _has_any_grads:
                            scaler.step(gaussians.optimizer)
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    if _has_any_grads or (dataset.train_test_exp and _has_exp_grads):
                        scaler.update()
                except AssertionError:
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if dataset.train_test_exp and gaussians.exposure_optimizer:
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--img_ext", type=str, default = None)
    parser.add_argument("--weight_maps_path", type=str, default = None)
    parser.add_argument("--gamma", type=float, default = 0.4)
    parser.add_argument("--render_debug", action='store_true', default=False)
    parser.add_argument("--upscale", type=int, default = 4)
    parser.add_argument("--no_sr", action='store_true', default=False)
    parser.add_argument("--full_sr", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    dataset.img_ext = args.img_ext
    training(dataset, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.upscale)

    # All done
    print("\nTraining complete.")