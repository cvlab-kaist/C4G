"""Motion supervision for the per-timestamp Gaussian sets of T4GS.

The encoder decodes the same learnable tokens once per timestamp, so token k
at timestamp t and token k at timestamp t' are the *same* Gaussian moved in
time. That correspondence is supervised with three terms:

    flow3d   ||(M_{t+}[k] - M_t[k]) - f_gt||   3D scene flow of the pixel the
             Gaussian projects to, from GT depth / 2D flow / next-frame depth
             unprojected with the batch cameras.
    flow2d   ||proj_{t+}(M_{t+}[k]) - (p_k + flow(p_k))||  2D flow re-projection
             (scale free, works without depth).
    static   ||M_t[k] - M_{t0}[k]||  for Gaussians that only ever land on
             static pixels: pin them to their first-timestamp position.
    flow3d_move / flow3d_still
             the flow3d residual split by whether the GT flow is above
             flow3d_move_rel (in scene-scale units) and averaged separately,
             each normalised to O(1): by |f_gt| for moving pairs, by the
             threshold for still ones. The pooled flow3d term is dominated by
             the many still pairs and, in the Huber linear regime, gives moving
             pairs a vanishing gradient; with the static anchor it drove every
             Gaussian to stop moving (scripts/diag_gaussian_motion.py).

Every view of the batch (context and target) can carry the maps described in
dataset/shims/motion_fields.py. Datasets without maps (EgoExo4D) get them
online from RAFT (2D flow, forward/backward checked) and the MoGe depth that the
depth supervision already computes.

Depth maps are aligned to the model's units with one per-batch median ratio
between the Gaussians' depths and the GT depths at their pixels: the DAVIS
intrinsics the model is fed differ from the true ones, so the model's depth is
a scaled version of the metric one even on GT-pose datasets.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MotionLossCfg:
    flow3d_weight: float = 0.0
    flow2d_weight: float = 0.0
    static_weight: float = 0.0
    flow3d_move_weight: float = 0.0
    flow3d_still_weight: float = 0.0
    # Absolute-error twin of flow3d_move: |f_pred - f_gt| / scene_scale on the moving pairs, L1
    # (flow3d_move_abs_huber_rel <= 0) or Huber with that delta. The relative term's quadratic regime
    # gives a gradient ~ (1 - alpha) / |f_gt| that fades near the target and for large motions, and
    # the v2 runs settle at 15-20 % of the GT displacement; a constant-magnitude gradient resists that.
    flow3d_move_abs_weight: float = 0.0
    flow3d_move_abs_huber_rel: float = 0.0
    # GT 3D flow above this fraction of the scene scale counts as motion.
    flow3d_move_rel: float = 0.005
    # Skip flow3d_move when fewer moving pairs than this support it.
    min_moving_pairs: int = 16
    # Linear ramp-in: zero before start_step, full after start_step + ramp_steps.
    start_step: int = 1000
    ramp_steps: int = 1000
    # A Gaussian counts as visible in a view when it lies within this relative
    # distance of the GT surface at its pixel (after scale alignment).
    visibility_rel_tol: float = 0.1
    # Huber transition points, relative to the batch scene scale / image size.
    huber_rel: float = 0.02
    huber_2d_rel: float = 0.01
    # Dilate the dynamic mask (pixels) before deciding the static token set.
    mask_dilation: int = 3
    # Stop gradients into the first-timestamp anchor of the static loss.
    detach_anchor: bool = False
    # Per-dataset multipliers on flow3d (e.g. lower for pseudo-GT depth).
    flow3d_dataset_weights: dict[str, float] = field(default_factory=dict)
    flow2d_dataset_weights: dict[str, float] = field(default_factory=dict)
    static_dataset_weights: dict[str, float] = field(default_factory=dict)
    flow3d_move_dataset_weights: dict[str, float] = field(default_factory=dict)
    flow3d_still_dataset_weights: dict[str, float] = field(default_factory=dict)
    flow3d_move_abs_dataset_weights: dict[str, float] = field(default_factory=dict)
    # Online pseudo-GT (RAFT flow + given depth) for these datasets, which ship
    # no maps; other datasets without maps only get the terms their maps allow.
    online_pseudo_gt: bool = True
    online_pseudo_gt_datasets: list[str] = field(default_factory=lambda: ["egoexo4d_mono"])
    online_flow_threshold: float = 0.5  # px at input resolution: dynamic if above
    online_fb_threshold: float = 1.0  # px: forward-backward consistency
    online_raft_iters: int = 12
    # Skip a term when fewer Gaussians than this support it.
    min_tokens: int = 32
    # The static anchor is only consistent with the (DAVIS-intrinsics) rendering
    # when the camera does not move; require that for these datasets.
    static_camera_only_datasets: list[str] = field(default_factory=lambda: ["spring"])
    static_camera_eps: float = 1e-3


def _project(means: Tensor, c2w: Tensor, intrinsics: Tensor, image_size: tuple[int, int]):
    """Project world points [N, 3] into one view -> pixels [N, 2], depth [N], in-bounds [N]."""
    H, W = image_size
    w2c = torch.linalg.inv(c2w)
    cam = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    fx, fy = intrinsics[0, 0] * W, intrinsics[1, 1] * H
    cx, cy = intrinsics[0, 2] * W, intrinsics[1, 2] * H
    zc = z.clamp(min=1e-6)
    px = cam[:, 0] / zc * fx + cx
    py = cam[:, 1] / zc * fy + cy
    p = torch.stack([px, py], dim=-1)
    inb = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return p, z, inb


def _unproject(p: Tensor, depth: Tensor, c2w: Tensor, intrinsics: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Pixels [N, 2] + z-depth [N] in one view -> world points [N, 3]."""
    H, W = image_size
    fx, fy = intrinsics[0, 0] * W, intrinsics[1, 1] * H
    cx, cy = intrinsics[0, 2] * W, intrinsics[1, 2] * H
    x = (p[:, 0] - cx) / fx * depth
    y = (p[:, 1] - cy) / fy * depth
    cam = torch.stack([x, y, depth], dim=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def _sample(img: Tensor, p: Tensor, mode: str = "nearest") -> Tensor:
    """Sample img [C, H, W] (or [H, W]) at pixel coords p [N, 2] (pixel i covers [i, i+1))."""
    squeeze = img.dim() == 2
    if squeeze:
        img = img[None]
    _, H, W = img.shape
    grid = torch.stack([2.0 * p[:, 0] / W - 1.0, 2.0 * p[:, 1] / H - 1.0], dim=-1)
    out = F.grid_sample(
        img[None].float(), grid[None, None], mode=mode, padding_mode="border", align_corners=False
    )[0, :, 0, :]  # [C, N]
    return out[0] if squeeze else out.T


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask[None, None].float(), k, stride=1, padding=radius)[0, 0]


def _huber(x: Tensor, delta: float) -> Tensor:
    return F.huber_loss(x, torch.zeros_like(x), delta=delta, reduction="none")


class MotionLoss(nn.Module):
    def __init__(self, cfg: MotionLossCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._raft: list[nn.Module] = []  # plain list: keep RAFT out of the state dict

    # ------------------------------------------------------------------
    # schedule / weights
    # ------------------------------------------------------------------

    def ramp(self, global_step: int) -> float:
        if global_step < self.cfg.start_step:
            return 0.0
        if self.cfg.ramp_steps <= 0:
            return 1.0
        return min(1.0, (global_step - self.cfg.start_step) / self.cfg.ramp_steps)

    def active(self, global_step: int) -> bool:
        weights = (
            self.cfg.flow3d_weight,
            self.cfg.flow2d_weight,
            self.cfg.static_weight,
            self.cfg.flow3d_move_weight,
            self.cfg.flow3d_still_weight,
            self.cfg.flow3d_move_abs_weight,
        )
        return any(w > 0 for w in weights) and self.ramp(global_step) > 0

    def term_weight(self, term: str, dataset_name: str, global_step: int) -> float:
        base = getattr(self.cfg, f"{term}_weight")
        mult = getattr(self.cfg, f"{term}_dataset_weights").get(dataset_name, 1.0)
        return base * mult * self.ramp(global_step)

    # ------------------------------------------------------------------
    # online pseudo ground truth (RAFT flow + provided depth)
    # ------------------------------------------------------------------

    def _get_raft(self, device: torch.device) -> nn.Module:
        if not self._raft:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._raft.append(model)
        model = self._raft[0]
        if next(model.parameters()).device != device:
            model.to(device)
        return model

    @torch.no_grad()
    def _raft_flow(self, img_a: Tensor, img_b: Tensor) -> Tensor:
        """RAFT flow a -> b for image batches [P, 3, H, W] in [0, 1] -> [P, H, W, 2] (dx, dy)."""
        model = self._get_raft(img_a.device)
        with torch.autocast(device_type="cuda", enabled=False):
            flows = model((img_a * 2 - 1).float(), (img_b * 2 - 1).float(), num_flow_updates=self.cfg.online_raft_iters)
        return flows[-1].permute(0, 2, 3, 1)

    @torch.no_grad()
    def build_online_pseudo_gt(
        self,
        views: dict,
        next_view: Tensor,
        depth: Tensor,
    ) -> dict:
        """Fill motion_* maps from RAFT (flow, validity, dynamic mask) and depth.

        depth: [V, H, W] per-view depth in a shared (arbitrary) scale.
        """
        image = views["image"]
        V, _, H, W = image.shape
        device = image.device
        src = [v for v in range(V) if int(next_view[v]) >= 0]
        flow = torch.zeros(V, H, W, 2, device=device)
        valid = torch.zeros(V, H, W, device=device)
        mask = torch.full((V, H, W), -1.0, device=device)
        depth_next = torch.zeros(V, H, W, device=device)
        motion_mag = torch.zeros(V, H, W, device=device)
        has_mag = torch.zeros(V, dtype=torch.bool, device=device)
        if src:
            dst = [int(next_view[v]) for v in src]
            a, b = image[src], image[dst]
            fwd = self._raft_flow(a, b)
            bwd = self._raft_flow(b, a)
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            base = torch.stack([xx + 0.5, yy + 0.5], dim=-1)  # [H, W, 2]
            for i, (v, vn) in enumerate(zip(src, dst)):
                f = fwd[i]
                target = base + f
                grid = torch.stack([2 * target[..., 0] / W - 1, 2 * target[..., 1] / H - 1], dim=-1)
                bwd_w = F.grid_sample(
                    bwd[i].permute(2, 0, 1)[None], grid[None], mode="bilinear", padding_mode="border", align_corners=False
                )[0].permute(1, 2, 0)
                inside = (grid.abs() <= 1).all(dim=-1)
                fb_err = (f + bwd_w).norm(dim=-1)
                ok = inside & (fb_err < self.cfg.online_fb_threshold)
                dn = F.grid_sample(depth[vn][None, None], grid[None], mode="nearest", padding_mode="border", align_corners=False)[0, 0]
                # Reject correspondences landing on depth discontinuities.
                dn_min = -F.max_pool2d(-depth[vn][None, None], 3, stride=1, padding=1)[0, 0]
                dn_max = F.max_pool2d(depth[vn][None, None], 3, stride=1, padding=1)[0, 0]
                spread = (dn_max - dn_min) / depth[vn].clamp(min=1e-6)
                spread_w = F.grid_sample(spread[None, None], grid[None], mode="nearest", padding_mode="border", align_corners=False)[0, 0]
                ok = ok & (depth[v] > 0) & (dn > 0) & (spread_w <= 0.05)
                mag = f.norm(dim=-1)
                dyn = mag > self.cfg.online_flow_threshold
                # Frame-to-frame scale jitter of the monocular depth: match the
                # next depth to the current one on static, consistent pixels.
                ref = ok & ~dyn
                if ref.sum() > 100:
                    ratio = (dn[ref] / depth[v][ref]).median()
                    if torch.isfinite(ratio) and ratio > 0:
                        dn = dn / ratio
                flow[v] = f
                valid[v] = ok.float()
                depth_next[v] = dn
                motion_mag[v] = mag
                has_mag[v] = True
                # The last view has no forward flow: use the backward one.
                if not has_mag[vn] and int(next_view[vn]) < 0:
                    motion_mag[vn] = bwd[i].norm(dim=-1)
                    has_mag[vn] = True
        for v in range(V):
            if has_mag[v]:
                mask[v] = (motion_mag[v] > self.cfg.online_flow_threshold).float()
        return {
            "depth": depth,
            "motion_flow": flow,
            "motion_depth_next": depth_next,
            "motion_mask": mask,
            "motion_valid": valid,
        }

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def forward(
        self,
        means_per_timestamp: dict[int, Tensor],  # {t: [N, 3]} world-space Gaussian means
        views: dict,  # one batch element: extrinsics [V,4,4] (c2w), intrinsics [V,3,3], index [V], camera [V], image [V,3,H,W], motion maps
        dataset_name: str,
        global_step: int,
        pseudo_depth: Optional[Tensor] = None,  # [V, H, W] monocular depth for online pseudo-GT
        depth_fn: Optional[Callable[[Tensor], Tensor]] = None,  # images [V,3,H,W] -> depth [V,H,W]
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        cfg = self.cfg
        c2w, K = views["extrinsics"], views["intrinsics"]
        index = views["index"]
        camera = views.get("camera")
        V = c2w.shape[0]
        H, W = views["image"].shape[-2:]
        device = c2w.device
        if camera is None:
            camera = torch.zeros(V, dtype=torch.long, device=device)

        losses: dict[str, Tensor] = {}
        stats: dict[str, float] = {}
        zero = torch.zeros((), device=device)

        timestamps = sorted(int(t) for t in torch.unique(index).tolist())
        if len(timestamps) < 2 or not all(t in means_per_timestamp for t in timestamps):
            return losses, stats
        t0 = timestamps[0]
        t_next_of = {t: timestamps[i + 1] for i, t in enumerate(timestamps[:-1])}

        # View that shows the same camera at the next timestamp (-1 if none).
        next_view = torch.full((V,), -1, dtype=torch.long, device=device)
        for v in range(V):
            t = int(index[v])
            if t not in t_next_of:
                continue
            hit = ((index == t_next_of[t]) & (camera == camera[v])).nonzero()
            if hit.numel() > 0:
                next_view[v] = hit[0, 0]

        # ---- supervision maps (GT from the loader, or online pseudo-GT) ----
        has_flow = views.get("motion_flow") is not None
        online = cfg.online_pseudo_gt and dataset_name in cfg.online_pseudo_gt_datasets
        if not has_flow and online and (pseudo_depth is not None or depth_fn is not None):
            if pseudo_depth is None:
                with torch.no_grad():
                    pseudo_depth = depth_fn(views["image"])
            pseudo_depth = torch.nan_to_num(pseudo_depth.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
            views = {**views, **self.build_online_pseudo_gt(views, next_view, pseudo_depth)}
            has_flow = True
            stats["online_pseudo_gt"] = 1.0
        depth_maps = views.get("depth")
        flow_maps = views.get("motion_flow")
        depth_next_maps = views.get("motion_depth_next")
        mask_maps = views.get("motion_mask")
        valid_maps = views.get("motion_valid")
        has_depth = depth_maps is not None
        has_mask = mask_maps is not None
        if not has_flow and not has_mask:
            return losses, stats

        # ---- project every timestamp's Gaussians into its views ----
        proj = []  # per view: dict(p, z, inb, d_gt, valid_gt)
        for v in range(V):
            t = int(index[v])
            p, z, inb = _project(means_per_timestamp[t], c2w[v], K[v], (H, W))
            entry = {"p": p, "z": z, "inb": inb}
            if has_depth:
                d = _sample(depth_maps[v], p.detach())
                entry["d_gt"] = d
                entry["valid_gt"] = inb & (d > 0)
            proj.append(entry)

        # ---- align GT depth to the model's units (one scalar per batch) ----
        scale = 1.0
        if has_depth:
            ratios = torch.cat([
                (e["z"][e["valid_gt"]] / e["d_gt"][e["valid_gt"]]).detach() for e in proj
            ])
            if ratios.numel() >= cfg.min_tokens:
                scale = float(ratios.median().clamp(min=1e-3, max=1e3))
        stats["depth_scale"] = scale

        # ---- visibility & scene scale ----
        vis = []
        for e in proj:
            if has_depth:
                d = e["d_gt"] * scale
                ok = e["valid_gt"] & ((e["z"] - d).abs() <= cfg.visibility_rel_tol * d)
            else:
                ok = e["inb"]
            vis.append(ok.detach())
        stats["visible_fraction"] = float(torch.stack([x.float().mean() for x in vis]).mean())
        if has_depth:
            ref = torch.cat([e["d_gt"][x] for e, x in zip(proj, vis)]) * scale
        else:
            ref = torch.cat([e["z"][x] for e, x in zip(proj, vis)])
        if ref.numel() < cfg.min_tokens:
            return losses, stats
        scene_scale = float(ref.detach().median().clamp(min=1e-4))
        stats["scene_scale"] = scene_scale

        # ---- flow losses ----
        if has_flow:
            f3_terms, f2_terms = [], []
            move_terms, still_terms, move_ratio, move_cos = [], [], [], []
            move_abs_terms = []
            split3d = cfg.flow3d_move_weight > 0 or cfg.flow3d_still_weight > 0 or cfg.flow3d_move_abs_weight > 0
            for v in range(V):
                vn = int(next_view[v])
                if vn < 0:
                    continue
                t, tn = int(index[v]), int(index[vn])
                e = proj[v]
                sel = vis[v]
                if valid_maps is not None:
                    sel = sel & (_sample(valid_maps[v], e["p"].detach()) > 0.5)
                if sel.sum() < cfg.min_tokens:
                    continue
                p = e["p"][sel]
                p_det = p.detach()
                f2 = _sample(flow_maps[v].permute(2, 0, 1), p_det)  # [n, 2]
                m_t = means_per_timestamp[t][sel]
                m_tn = means_per_timestamp[tn][sel]
                if cfg.flow2d_weight > 0:
                    pn, zn, _ = _project(m_tn, c2w[vn], K[vn], (H, W))
                    front = zn > 1e-4
                    err2 = (pn - (p_det + f2)) / max(H, W)
                    l2 = _huber(err2, cfg.huber_2d_rel).sum(-1)[front]
                    if l2.numel() > 0:
                        f2_terms.append(l2)
                if (cfg.flow3d_weight > 0 or split3d) and has_depth and depth_next_maps is not None:
                    d = e["d_gt"][sel] * scale
                    dn = _sample(depth_next_maps[v], p_det) * scale
                    ok = dn > 0
                    X = _unproject(p_det, d, c2w[v], K[v], (H, W))
                    Xn = _unproject(p_det + f2, dn, c2w[vn], K[vn], (H, W))
                    f_gt = (Xn - X)[ok]
                    f_pred = (m_tn - m_t)[ok]
                    err3 = (f_pred - f_gt) / scene_scale
                    if cfg.flow3d_weight > 0:
                        l3 = _huber(err3, cfg.huber_rel).sum(-1)
                        if l3.numel() > 0:
                            f3_terms.append(l3)
                    if split3d and err3.shape[0] > 0:
                        gt_mag = (f_gt / scene_scale).norm(dim=-1)
                        moving = gt_mag > cfg.flow3d_move_rel
                        err_mag = err3.norm(dim=-1)
                        if moving.any():
                            # Relative end-point error: 1 when the Gaussian does not move at all.
                            move_terms.append(_huber(err_mag[moving] / gt_mag[moving], 1.0))
                            move_abs_terms.append(err_mag[moving] if cfg.flow3d_move_abs_huber_rel <= 0
                                                  else _huber(err_mag[moving], cfg.flow3d_move_abs_huber_rel))
                            pred = f_pred[moving].detach()
                            move_ratio.append(pred.norm(dim=-1) / f_gt[moving].norm(dim=-1).clamp(min=1e-12))
                            move_cos.append(F.cosine_similarity(pred, f_gt[moving], dim=-1))
                        if (~moving).any():
                            still_terms.append(_huber(err_mag[~moving] / cfg.flow3d_move_rel, 1.0))
            if f3_terms:
                all3 = torch.cat(f3_terms)
                losses["flow3d"] = all3.mean()
                stats["flow3d_tokens"] = float(all3.numel())
            if f2_terms:
                all2 = torch.cat(f2_terms)
                losses["flow2d"] = all2.mean()
                stats["flow2d_tokens"] = float(all2.numel())
            if move_terms:
                all_move = torch.cat(move_terms)
                stats["flow3d_moving_pairs"] = float(all_move.numel())
                # Motion health, independent of the weights: ~0 ratio / cos means frozen Gaussians.
                stats["flow3d_move_ratio"] = float(torch.cat(move_ratio).median())
                stats["flow3d_move_cos"] = float(torch.cat(move_cos).median())
                if cfg.flow3d_move_weight > 0 and all_move.numel() >= cfg.min_moving_pairs:
                    losses["flow3d_move"] = all_move.mean()
                if cfg.flow3d_move_abs_weight > 0 and all_move.numel() >= cfg.min_moving_pairs:
                    losses["flow3d_move_abs"] = torch.cat(move_abs_terms).mean()
            if still_terms and cfg.flow3d_still_weight > 0:
                all_still = torch.cat(still_terms)
                if all_still.numel() >= cfg.min_tokens:
                    losses["flow3d_still"] = all_still.mean()

        # ---- static anchor ----
        cam_t = c2w[:, :3, 3]
        camera_static = bool((cam_t - cam_t[0:1]).norm(dim=-1).max() < cfg.static_camera_eps)
        stats["camera_static"] = float(camera_static)
        static_allowed = camera_static or dataset_name not in cfg.static_camera_only_datasets
        if cfg.static_weight > 0 and has_mask and static_allowed:
            N = means_per_timestamp[t0].shape[0]
            seen = torch.zeros(N, dtype=torch.bool, device=device)
            static_ok = torch.ones(N, dtype=torch.bool, device=device)
            for v in range(V):
                m = mask_maps[v]
                unknown = m < 0
                dyn = _dilate((m > 0.5).float(), cfg.mask_dilation) > 0.5
                bad = dyn | unknown
                hit = _sample(bad.float(), proj[v]["p"].detach()) > 0.5
                seen |= vis[v]
                static_ok &= ~(vis[v] & hit)
            S = seen & static_ok
            stats["static_fraction"] = float(S.float().mean())
            if S.sum() >= cfg.min_tokens:
                anchor = means_per_timestamp[t0][S]
                if cfg.detach_anchor:
                    anchor = anchor.detach()
                terms = []
                for t in timestamps[1:]:
                    err = (means_per_timestamp[t][S] - anchor) / scene_scale
                    terms.append(_huber(err, cfg.huber_rel).sum(-1))
                losses["static"] = torch.cat(terms).mean()

        return losses, stats
