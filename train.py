# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import platform
from statistics import mean, median

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_llff import DatasetLLFF

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from geometry.flexicubes_geo import FlexiCubesGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

# Intrinsic pipeline (ordinal albedo)
try:
    from intrinsic.pipeline import load_models, run_gray_pipeline
    INTRINSIC_AVAILABLE = True
except ImportError:
    INTRINSIC_AVAILABLE = False
    print("Warning: Intrinsic module not available")


RADIUS = 6.0

# torch.autograd.set_detect_anomaly(True)

# -----------------------------------------------------------------------------


class BenchLogger:
    def __init__(self, out_dir, flags):
        self.enabled = getattr(flags, 'bench', False) and flags.local_rank == 0
        self.rows = []
        self.flags = flags
        self.out_dir = out_dir
        if not self.enabled:
            return
        self.jsonl_path = os.path.join(out_dir, flags.bench_output)
        self.summary_path = os.path.join(out_dir, flags.bench_summary)
        with open(self.jsonl_path, 'w') as f:
            f.write('')  # truncate

    def _sysinfo(self):
        dev_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_id)
        try:
            nvdiffrast_ver = getattr(dr, '__version__', 'unknown')
        except:
            nvdiffrast_ver = 'unknown'
        return {
            "device_index": dev_id,
            "device_name": props.name,
            "sm_count": props.multi_processor_count,
            "total_mem_gb": round(props.total_memory / (1024**3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "nvdiffrast_version": nvdiffrast_ver,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

    def log(self, row: dict):
        if not self.enabled:
            return
        self.rows.append(row)
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    def summarize(self):
        if not self.enabled or len(self.rows) == 0:
            return

        def agg(key):
            vals = [r[key] for r in self.rows if key in r]
            if not vals:
                return None
            return {
                "count": len(vals),
                "mean": float(mean(vals)),
                "median": float(median(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }

        summary = {
            "sysinfo": self._sysinfo(),
            "flags": {k: self.flags.__dict__[k] for k in self.flags.__dict__},
            "metrics": {
                "iter_ms": agg("iter_ms"),
                "data_ms": agg("data_ms"),
                "fwd_ms": agg("fwd_ms"),
                "bwd_ms": agg("bwd_ms"),
                "opt_ms": agg("opt_ms"),
                "clamp_ms": agg("clamp_ms"),
                "throughput_mpix_s": agg("throughput_mpix_s"),
                "images_per_s": agg("images_per_s"),
                "max_mem_gb": agg("max_mem_gb"),
                "reserved_gb": agg("reserved_gb"),
                "img_loss": agg("img_loss"),
                "reg_loss": agg("reg_loss"),
                "intr_loss": agg("intr_loss"),
                "lr": agg("lr"),
            }
        }
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        it = summary["metrics"]["iter_ms"]
        tp = summary["metrics"]["throughput_mpix_s"]
        if it and tp:
            print(f'[BENCH] median iter {it["median"]:.2f} ms | '
                  f'{tp["median"]:.1f} MPix/s | '
                  f'{summary["sysinfo"]["device_name"]} '
                  f'({summary["sysinfo"]["compute_capability"]})')


@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False


@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8),
                                  dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, f"Unknown background type {bg_type}"

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    target['img'] = torch.cat(
        (torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]),
         target['img'][..., 3:4]), dim=-1)

    return target

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)

    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])

    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[..., 0:1])), dim=-1)

    kd_min = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda')
    kd_max = torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda')
    ks_max = torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda')
    nrm_max = torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf': mat['bsdf'],
        'kd': texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks': texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal': texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })
    return new_mesh


def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda')
    kd_max = torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda')
    ks_max = torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda')
    nrm_max = torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat = material.Material({'kd_ks_normal': mlp_map_opt})
    else:
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * \
                      (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init, FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].item(), high=ks_max[1].item())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].item(), high=ks_max[2].item())
            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2),
                                                  FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({'kd': kd_map_opt, 'ks': ks_map_opt, 'normal': normal_map_opt})

    mat['bsdf'] = init_mat['bsdf'] if init_mat is not None else 'pbr'
    return mat


@torch.no_grad()
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    lgt.build_mips()
    if FLAGS.camera_space_light:
        lgt.xfm(target['mv'])

    buffers = geometry.render(glctx, target, lgt, opt_material)
    result_dict['ref'] = util.rgb_to_srgb(target['img'][..., 0:3])[0]
    result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][..., 0:3])[0]
    result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

    if FLAGS.display is not None:
        for layer in FLAGS.display:
            if 'latlong' in layer and layer['latlong']:
                if isinstance(lgt, light.EnvironmentLight):
                    result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
            elif 'relight' in layer:
                if not isinstance(layer['relight'], light.EnvironmentLight):
                    layer['relight'] = light.load_env(layer['relight'])
                img = geometry.render(glctx, target, layer['relight'], opt_material)
                rgb = img['shaded'][..., 0:3]
                result_dict['relight'] = util.rgb_to_srgb(rgb)[0]
                result_image = torch.cat([result_image, result_dict['relight']], axis=1)
            elif 'bsdf' in layer:
                buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                if layer['bsdf'] == 'kd':
                    result_dict['kd'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                    result_image = torch.cat([result_image, result_dict['kd']], axis=1)
                elif layer['bsdf'] == 'normal':
                    n = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    result_image = torch.cat([result_image, n], axis=1)
                else:
                    result_image = torch.cat([result_image, buffers['shaded'][0, ..., 0:3]], axis=1)

    return result_image, result_dict


@torch.no_grad()
def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    os.makedirs(out_dir, exist_ok=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    print("Running validation")

    def mse(a, b):
        return torch.mean((a - b) ** 2).item()

    def psnr(a, b, eps=1e-8):
        m = mse(a, b)
        return 20.0 * np.log10(1.0) - 10.0 * np.log10(max(m, eps))

    mse_values, psnr_values = [], []

    for it, target in enumerate(dataloader_validate):
        target = prepare_batch(target, FLAGS.background)

        res_img, _ = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
        opt = res_img[:, :res_img.shape[1] // 2, ...]  # left half
        ref = res_img[:, res_img.shape[1] // 2:, ...]  # right half

        mse_values.append(mse(opt, ref))
        psnr_values.append(psnr(opt, ref))

        if FLAGS.save_interval and (it % FLAGS.save_interval == 0):
            util.save_image(os.path.join(out_dir, f'val_{it:06d}.png'),
                            res_img.detach().cpu().numpy())

    avg_mse = float(np.mean(mse_values)) if mse_values else float('nan')
    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float('nan')

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write('ID, MSE, PSNR\n')
        f.write(f'AVERAGES: {avg_mse:1.6f}, {avg_psnr:2.3f}\n')

    print("Validation done. AVERAGES -> MSE: %.6f, PSNR: %.3f dB" % (avg_mse, avg_psnr))
    return avg_psnr


class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()
        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        # Intrinsic loss control
        self.use_intrinsic = getattr(FLAGS, 'use_intrinsic', False) and INTRINSIC_AVAILABLE
        self.lambda_intr = getattr(FLAGS, 'intrinsic_lambda', 0.1)
        self.intrinsic_every = getattr(FLAGS, 'intrinsic_every', 10)
        self.scale_mode = getattr(FLAGS, 'intrinsic_scale_mode', 'avg')  # ['avg', 'none']
        
        # Load intrinsic models if enabled
        self.ordinal_models = None
        if self.use_intrinsic:
            try:
                print("Loading intrinsic models (v2)...")
                self.ordinal_models = load_models('v2', stage=4, device='cuda')
                print("Intrinsic models loaded successfully")
            except Exception as e:
                print(f"Failed to load intrinsic models: {e}")
                self.use_intrinsic = False

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        # Baseline tick (renders + returns image and regularization losses)
        img_loss, reg_loss = self.geometry.tick(self.glctx, target, self.light, self.material, self.image_loss_fn, it)

        # Initialize intrinsic loss for this iteration
        self.last_intr_loss = torch.tensor(0.0, device=img_loss.device)
        
        # Intrinsic branch (every N steps)
        do_intrinsic = self.use_intrinsic and (self.intrinsic_every > 0) and (it % self.intrinsic_every == 0)
        if do_intrinsic:
            try:
                assert self.ordinal_models is not None, "Intrinsic enabled but ordinal models not loaded."

                # Render KD (diffuse-only) buffers now
                kd_buffers = self.geometry.render(self.glctx, target, self.light, self.material, bsdf='kd')
                kd_rgba = kd_buffers['shaded']                     # [B,H,W,4]
                mask = kd_rgba[..., 3:].permute(0, 3, 1, 2)        # [B,1,H,W]

                pred_list = []
                
                for img_tensor in kd_rgba[..., :3]:                # [H,W,3] (GPU)
                    img_np = img_tensor.detach().cpu().numpy()    # HWC numpy
                    h, w = img_np.shape[:2]
                    base = min(h, w)

                    # Use run_gray_pipeline for full albedo estimation
                    result = run_gray_pipeline(
                        self.ordinal_models, 
                        img_np, 
                        base_size=base,
                        maintain_size=True,
                        linear=False,
                        device='cuda'
                    )
                    
                    # Use 'hr_alb' if available (stage >= 3), else fallback to 'gry_alb' 
                    if 'hr_alb' in result:
                        alb_np = result['hr_alb']  # (H,W,3)
                    elif 'lr_alb' in result:
                        alb_np = result['lr_alb']  # (H,W,3)
                    elif 'gry_alb' in result:
                        alb_np = result['gry_alb']  # (H,W,3) - grayscale only
                    else:
                        continue
                    
                    alb_pred = torch.from_numpy(alb_np).permute(2, 0, 1).to(img_tensor.device)  # 3,H,W

                    # match resolution if needed
                    if (alb_pred.shape[1], alb_pred.shape[2]) != (h, w):
                        alb_pred = F.interpolate(alb_pred.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]

                    pred_list.append(alb_pred)

                if len(pred_list) > 0:
                    pred_alb = torch.stack(pred_list, 0)                            # [B,3,H,W]
                    cur_kd = kd_rgba[..., :3].permute(0, 3, 1, 2)                  # [B,3,H,W]
                    intr_loss = F.l1_loss(pred_alb * mask, cur_kd * mask)

                    scale = (self.intrinsic_every if self.scale_mode == 'avg' else 1.0)
                    reg_loss = reg_loss + (self.lambda_intr * scale) * intr_loss
                    self.last_intr_loss = intr_loss.detach()

                    # optional debug dump
                    if self.FLAGS.local_rank == 0 and self.FLAGS.save_interval and (it % self.FLAGS.save_interval == 0):
                        kd0 = cur_kd[0].permute(1, 2, 0).clamp(0, 1)      # HWC torch
                        alb0 = pred_alb[0].permute(1, 2, 0).clamp(0, 1)   # HWC torch
                        util.save_image(f"{self.FLAGS.out_dir}/intr_kd_input_{it:06d}.png",
                                        util.rgb_to_srgb(kd0).cpu().numpy())
                        util.save_image(f"{self.FLAGS.out_dir}/intr_albedo_pred_{it:06d}.png",
                                        util.rgb_to_srgb(alb0).cpu().numpy())
            except Exception as e:
                if it == 0:
                    print(f"WARNING: intrinsic loss failed: {e}")

        return img_loss, reg_loss


def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True,
    bench=None,
    base_mesh=None
):

    # Learning rates (pos, material) if tuple/list, else scalar
    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, (list, tuple)) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, (list, tuple)) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, (list, tuple)) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(0.0, 10 ** (-(iter - warmup_iter) * 0.0002))

    image_loss_fn = createLoss(FLAGS)
    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS)
    trainer_noddp.use_intrinsic = FLAGS.use_intrinsic
    trainer_noddp.lambda_intr = FLAGS.intrinsic_lambda

    # Load ordinal models ONCE if intrinsic is enabled
    if FLAGS.use_intrinsic:
        try:
            from intrinsic.pipeline import load_models
            trainer_noddp.ordinal_models = load_models('v2', device='cuda')
        except ImportError:
            print("WARNING: intrinsic module not available, disabling intrinsic loss")
            trainer_noddp.use_intrinsic = False

    if FLAGS.isosurface == 'flexicubes':
        betas = (0.7, 0.9)
    else:
        betas = (0.9, 0.999)

    if FLAGS.multi_gpu:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))
        optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))
    else:
        trainer = trainer_noddp
        if optimize_geometry:
            optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))
        optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    img_cnt = 0
    img_loss_vec, reg_loss_vec, iter_dur_vec, intr_loss_vec = [], [], [], []

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):
        wall_iter_t0 = time.perf_counter()

        # Data prep
        data_t0 = time.perf_counter()
        target = prepare_batch(target, 'random')
        data_ms = (time.perf_counter() - data_t0) * 1000.0

        # Display/save BEFORE step
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS)
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)

                    # Save Kd safely (2D kd if present; else try result_dict)
                    if 'kd' in opt_material:
                        texture.save_texture2D(FLAGS.out_dir + '/' + ('img_%s_%06d_kd.png' % (pass_name, img_cnt)),
                                               texture.rgb_to_srgb(opt_material['kd']))
                    else:
                        if 'kd' in result_dict:
                            kd_img = result_dict['kd'].detach().cpu().numpy()
                            util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d_kd.png' % (pass_name, img_cnt)), kd_img)
                    img_cnt += 1

        # CUDA timing
        torch.cuda.reset_peak_memory_stats()
        ev_fwd_s = torch.cuda.Event(enable_timing=True)
        ev_fwd_e = torch.cuda.Event(enable_timing=True)
        ev_bwd_s = torch.cuda.Event(enable_timing=True)
        ev_bwd_e = torch.cuda.Event(enable_timing=True)
        ev_opt_s = torch.cuda.Event(enable_timing=True)
        ev_opt_e = torch.cuda.Event(enable_timing=True)
        ev_clp_s = torch.cuda.Event(enable_timing=True)
        ev_clp_e = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        ev_fwd_s.record()
        img_loss, reg_loss = trainer(target, it)
        intr_loss_vec.append(trainer.last_intr_loss.item())
        ev_fwd_e.record()

        total_loss = img_loss + reg_loss
        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        ev_bwd_s.record()
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material and opt_material['kd_ks_normal'].encoder is not None:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0
        ev_bwd_e.record()

        ev_opt_s.record()
        optimizer.step()
        scheduler.step()
        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()
        ev_opt_e.record()

        ev_clp_s.record()
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)
        ev_clp_e.record()

        torch.cuda.current_stream().synchronize()

        fwd_ms = ev_fwd_s.elapsed_time(ev_fwd_e)
        bwd_ms = ev_bwd_s.elapsed_time(ev_bwd_e)
        opt_ms = ev_opt_s.elapsed_time(ev_opt_e)
        clp_ms = ev_clp_s.elapsed_time(ev_clp_e)
        iter_ms = (time.perf_counter() - wall_iter_t0) * 1000.0
        iter_dur_vec.append(iter_ms)

        if bench is not None and bench.enabled and it >= FLAGS.bench_start:
            H, W = FLAGS.train_res
            pixels = FLAGS.batch * H * W
            seconds = max(iter_ms / 1000.0, 1e-6)
            throughput_mpix_s = (pixels / 1e6) / seconds
            images_per_s = FLAGS.batch / seconds
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            bench.log({
                "iter": it,
                "img_loss": float(img_loss.item()),
                "reg_loss": float(reg_loss.item()),
                "intr_loss": float(trainer.last_intr_loss.item()),
                "lr": float(optimizer.param_groups[0]['lr']),
                "iter_ms": float(iter_ms),
                "data_ms": float(data_ms),
                "fwd_ms": float(fwd_ms),
                "bwd_ms": float(bwd_ms),
                "opt_ms": float(opt_ms),
                "clamp_ms": float(clp_ms),
                "throughput_mpix_s": float(throughput_mpix_s),
                "images_per_s": float(images_per_s),
                "max_mem_gb": float(max_mem_gb),
                "reserved_gb": float(reserved_gb),
                "pass": pass_name,
            })

        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            intr_loss_avg = np.mean(intr_loss_vec[-log_interval:]) if intr_loss_vec else 0.0
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            remaining_time = (FLAGS.iter - it) * (iter_dur_avg / 1000.0)

            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, intrinsc_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" %
                  (it, img_loss_avg, reg_loss_avg, intr_loss_avg,
                   optimizer.param_groups[0]['lr'], iter_dur_avg,
                   util.time_to_text(remaining_time)))

    if bench is not None:
        bench.summarize()

    return geometry, opt_material

# ---------  -----------------------------------------------------------------------
# Main
# -------- -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)  # or [pos,mat]
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--isosurface', default='dmtet', choices=['dmtet', 'flexicubes'])

    # Intrinsic ablations
    parser.add_argument('-intr-loss', '--use-intrinsic', action='store_true',
                        help='Enable intrinsic-image (ordinal albedo) L1 loss')
    parser.add_argument('-intr-lambda', '--intrinsic-lambda', type=float, default=0.1,
                        help='Weight for intrinsic loss when enabled')
    parser.add_argument('--intrinsic-every', type=int, default=10,
                        help='Compute intrinsic loss every N iterations (default: 10)')
    parser.add_argument('--intrinsic-scale-mode', choices=['avg', 'none'], default='avg',
                        help='avg: multiply by N to keep average strength unchanged; none: no scaling')

    # Bench
    parser.add_argument('--bench', action='store_true', default=False,
                        help='Enable per-iter benchmarking & dump JSONL to out_dir')
    parser.add_argument('--bench-start', type=int, default=100,
                        help='Start collecting benchmark rows at this iteration')
    parser.add_argument('--bench-output', type=str, default='bench.jsonl',
                        help='Per-iter JSONL filename (inside out_dir)')
    parser.add_argument('--bench-summary', type=str, default='bench_summary.json',
                        help='Final summary JSON filename (inside out_dir)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')

    FLAGS = parser.parse_args()

    # Defaults / derived
    FLAGS.mtl_override = None
    FLAGS.dmtet_grid = 64
    FLAGS.mesh_scale = 2.1
    FLAGS.env_scale = 1.0
    FLAGS.envmap = None
    FLAGS.display = [{'bsdf': 'kd'}]
    FLAGS.camera_space_light = False
    FLAGS.lock_light = False
    FLAGS.lock_pos = False
    FLAGS.sdf_regularizer = 0.2
    FLAGS.laplace = "relative"
    FLAGS.laplace_scale = 10000.0
    FLAGS.pre_load = True
    FLAGS.kd_min = [0.0, 0.0, 0.0, 0.0]
    FLAGS.kd_max = [1.0, 1.0, 1.0, 1.0]
    FLAGS.ks_min = [0.0, 0.08, 0.0]
    FLAGS.ks_max = [1.0, 1.0, 1.0]
    FLAGS.nrm_min = [-1.0, -1.0, 0.0]
    FLAGS.nrm_max = [1.0, 1.0, 1.0]
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.learn_light = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "0")) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'
        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = f'out/cube_{FLAGS.train_res}'
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir if not FLAGS.out_dir.startswith('out/') else FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)

    bench = BenchLogger(FLAGS.out_dir, FLAGS)
    glctx = dr.RasterizeGLContext()

    # Data pipeline
    if os.path.splitext(FLAGS.ref_mesh)[1] == '.obj':
        ref_mesh = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)

        bbox_min = ref_mesh.v_pos.min(0).values
        bbox_max = ref_mesh.v_pos.max(0).values
        diag = (bbox_max - bbox_min).norm().item()
        RADIUS = max(2.0, diag * 2.0)
        print(f"Auto-radius set to {RADIUS:.2f}")

        dataset_train = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
        dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)
    elif os.path.isdir(FLAGS.ref_mesh):
        if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
            dataset_train = DatasetLLFF(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter + 1) * FLAGS.batch)
            dataset_validate = DatasetLLFF(FLAGS.ref_mesh, FLAGS)
        elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json')):
            dataset_train = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, examples=(FLAGS.iter + 1) * FLAGS.batch)
            dataset_validate = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)
        else:
            raise RuntimeError("Unknown dataset layout for ref_mesh")
    else:
        raise RuntimeError(f"ref_mesh path not found: {FLAGS.ref_mesh}")

    # Light
    FLAGS.learn_light = True
    FLAGS.lock_pos = True
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    if FLAGS.base_mesh is None:
        # Pass 1 – SDF isosurface
        if FLAGS.isosurface == 'flexicubes':
            geometry = FlexiCubesGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        elif FLAGS.isosurface == 'dmtet':
            geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        else:
            raise ValueError(f"Invalid isosurface {FLAGS.isosurface}")

        mat = initial_guess_material(geometry, True, FLAGS)

        geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate,
                                      FLAGS, pass_idx=0, pass_name="dmtet_pass1",
                                      optimize_light=FLAGS.learn_light, bench=bench, base_mesh=None)

        if FLAGS.local_rank == 0 and FLAGS.validate:
            validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

        base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

        # cleanup MLP tex
        torch.cuda.empty_cache()
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']

        lgt = lgt.clone()
        geometry = DLMesh(base_mesh, FLAGS)

        if FLAGS.local_rank == 0:
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
            obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
            light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

        # Pass 2 – fixed topology
        geometry, mat = optimize_mesh(glctx, geometry, base_mesh.material, lgt,
                                      dataset_train, dataset_validate, FLAGS,
                                      pass_idx=1, pass_name="mesh_pass", warmup_iter=100,
                                      optimize_light=FLAGS.learn_light and not FLAGS.lock_light,
                                      optimize_geometry=not FLAGS.lock_pos, bench=bench, base_mesh=base_mesh)
    else:
        # Single pass – fixed topology
        base_mesh = mesh.load_mesh(FLAGS.base_mesh)
        center = base_mesh.v_pos.mean(dim=0, keepdim=True)
        base_mesh.v_pos -= center

        geometry = DLMesh(base_mesh, FLAGS)
        mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)

        geometry, mat = optimize_mesh(glctx, geometry, mat, lgt,
                                      dataset_train, dataset_validate, FLAGS,
                                      pass_idx=0, pass_name="mesh_pass",
                                      warmup_iter=100, optimize_light=FLAGS.learn_light,
                                      optimize_geometry=not FLAGS.lock_pos, bench=bench, base_mesh=base_mesh)

    # Final validate + dump
    if FLAGS.validate and FLAGS.local_rank == 0:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS)

    if FLAGS.local_rank == 0:
        final_mesh = geometry.getMesh(mat)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)
