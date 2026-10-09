"""
Edge-matching pose-refinement eval — GENERALIZED to match eval_full.py config handling.

Per test frame:
  1. Run LEAR ONCE -> flow_up + depth2edge (single forward pass; no second run).
  2. BASELINE pose = Flow2Pose(flow_up, depth) ; error vs the perturbation.
  3. REFINEMENT: 3D edge source = the model's depth2edge prediction, deprojected through the
     input depth -> 3D points X; optimise the 6-DOF pose (init = LEAR's baseline) to align the
     projection of X with the observed EVENT edges (chamfer / distance-transform, LBFGS).
  4. REFINED pose error vs the same perturbation.

Geometry (crop / max_depth / ev_input) is auto from core/scene_config.py per --seq, exactly like
eval_full.py. Supports --dataset {m3ed,dsec}, --test_rt (test perturbation files), --iters,
and --dense_depth/--partial_depth (feed ch1 to match how the checkpoint was trained).

GT pose is used only to SCORE (err_Pose) — never to build the edges or the init, same as main.py.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.append("core")
from core.datasets_m3ed import DatasetM3ED
from core.datasets_dsec import DatasetDSEC
from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose, err_Pose
from core.utils_point import quat2mat, quaternion_from_matrix
from core.scene_config import scene_geometry, scene_ev_input
from core.model_builder import build_model
from tools.pose_refine.edge_extract import skeletonize
from tools.pose_refine.refine import (distance_transform, deproject, project,
                                      exp_so3, refine_pose, sample_dt)


# ---- se(3) <-> Flow2Pose (quaternion + flag flip) -------------------------
def flip_quat(q):
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def so3_log(R):
    cos = ((R[0, 0] + R[1, 1] + R[2, 2] - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = torch.sin(theta)
    scale = torch.where(s.abs() < 1e-6, torch.full_like(theta, 0.5), theta / (2 * s))
    return scale * w


def lear_to_xi(R_pred, T_pred):
    """Flow2Pose (post-flag-flip) quaternion+t -> se(3) xi=(omega,t) of the RAW PnP pose."""
    q_raw = flip_quat(R_pred)
    R_raw = quat2mat(q_raw)[:3, :3]
    t_raw = -T_pred
    return torch.cat([so3_log(R_raw), t_raw])


def xi_to_lear(xi):
    """se(3) xi -> Flow2Pose convention (R_pred quaternion, T_pred)."""
    R = exp_so3(xi[:3])
    q = quaternion_from_matrix(R)                     # [w,x,y,z]
    return flip_quat(q), -xi[3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", "--test_sequence", dest="seq", default="falcon_indoor_flight_3")
    ap.add_argument("--dataset", default="m3ed", choices=["m3ed", "dsec"])
    ap.add_argument("--data_path", default=None, help="override data root (default per dataset)")
    ap.add_argument("--ev_input", default=None, help="override event rep dir (default: scene_config per seq)")
    ap.add_argument("--max_depth", type=float, default=None, help="override max_depth (default: scene_config)")
    ap.add_argument("--load_checkpoints", required=True, help="the LEAR checkpoint to refine from")
    ap.add_argument("--test_rt", default=None, help="explicit test_RT csv path; default = the shipped per-sequence perturbation file")
    ap.add_argument("--iters", "--test_iters", dest="iters", type=int, default=24, help="IFR iterations at inference")
    ap.add_argument("--dense_depth", action="store_true", help="feed dense ch1 (match training)")
    ap.add_argument("--partial_depth", default=None, choices=["partial_light"],  # paper's partial completion; 'partial'/'partial_dc' disabled (exploratory)
                    help="feed a partial-completion ch1 (match training); implies reading ch1")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--edge_thresh", type=float, default=0.5)
    ap.add_argument("--skeleton_pred", action="store_true")
    ap.add_argument("--no_skeleton_target", action="store_true")
    ap.add_argument("--refine_iters", type=int, default=80)
    ap.add_argument("--huber", type=float, default=3.0)
    ap.add_argument("--zreg", type=float, default=1e-3)
    ap.add_argument("--anchor_w", type=float, default=0.0, help="trust-region prior on ||xi-init||^2")
    ap.add_argument("--lock_rot", action="store_true", help="translation-only refinement (freeze rotation at LEAR init)")
    ap.add_argument("--init_poses", default=None,
                    help="npz of saved per-frame poses (eval_full --save_poses / iter_sweep) to use as the refinement "
                         "init, so the baseline == our reported pose. Selected by --select. If unset, uses Flow2Pose (all).")
    ap.add_argument("--select", default="median", choices=["all", "median", "floor0", "floor0.5"],
                    help="which saved pose to refine from --init_poses: 'median'=conf>median(top50) [our headline], "
                         "'floor0'/'floor0.5'=conf pweight(fl0)/(fl0.5) [pweight selections], 'all'.")
    ap.add_argument("--min_pts", type=int, default=30)
    ap.add_argument("--edge_subsample", type=int, default=1,
                    help="keep 1 of every N predicted edge pixels (stride) to speed up the LM; 1=all (chamfer is over-determined)")
    ap.add_argument("--sweep_both", action="store_true",
                    help="AMORTIZED: forward once/frame, run BOTH-weighting over {full,transl}x{aw10,30,300}, write 6 CSVs "
                         "(the forward is the bottleneck; this reuses it across the 6 configs). Needs --out_prefix.")
    ap.add_argument("--sweep_src", default="both", choices=["both", "edgeprob", "flow"],
                    help="edge weighting used by --sweep_both: both=edgeprob*conf (default), edgeprob only, or flow conf only; "
                         "CSV names use the source: <prefix>__<src>__<op>__aw<K>.csv")
    ap.add_argument("--sweep_aws", default="10,30,300",
                    help="comma-separated anchor weights for --sweep_both (each run with full and transl); default 10,30,300")
    ap.add_argument("--out_prefix", default=None, help="output prefix for --sweep_both: writes <prefix>__both__<op>__aw<K>.csv")
    ap.add_argument("--edge_conf", default="none", choices=["none", "weight", "pweight"],
                    help="use the per-pixel confidence (same head as correspondence selection) on the EDGE points: "
                         "'weight'=weighted chamfer cost (w_i=conf); 'pweight'=probabilistically keep each edge pixel "
                         "w.p. floor+(1-floor)*conf_norm then uniform-refine the subset; 'none'=uniform (default).")
    ap.add_argument("--edge_conf_floor", type=float, default=0.5, help="floor for --edge_conf pweight (default 0.5)")
    ap.add_argument("--edge_conf_seed", type=int, default=0, help="RNG seed for --edge_conf pweight sampling")
    ap.add_argument("--edge_conf_src", default="flow", choices=["flow", "edgeprob", "both"],
                    help="source of the per-edge confidence: 'flow'=the dense flow-correspondence head (same as selection); "
                         "'edgeprob'=the model's depth2edge edge-existence probability (intrinsic edge reliability); "
                         "'both'=edgeprob * flow-confidence (edge is real AND its flow/geometry is reliable).")
    ap.add_argument("--out", default="pose_refine_eval.csv")
    args = ap.parse_args()
    edge_rng = np.random.default_rng(args.edge_conf_seed)
    _aws = [int(float(x)) for x in args.sweep_aws.split(",")]
    SWEEP_CFGS = [("full", aw) for aw in _aws] + [("transl", aw) for aw in _aws]   # default == the original 6 configs
    if args.sweep_both:
        args.edge_conf = "weight"; args.edge_conf_src = args.sweep_src   # default both = edgeprob * conf, config-independent

    # geometry / ev auto from scene_config (same source of truth as eval_full.py)
    g = scene_geometry(args.seq, args.dataset)
    CROP_H, CROP_W, CROP_X, CROP_Y = g["crop"]
    MAXD = args.max_depth if args.max_depth is not None else g["max_depth"]
    ev = args.ev_input or scene_ev_input(args.seq, args.dataset)
    ch1 = args.dense_depth or (args.partial_depth is not None)
    print(f"[cfg] seq={args.seq} dataset={args.dataset} class={g['klass']} ev={ev} "
          f"crop=({CROP_H},{CROP_W},{CROP_X},{CROP_Y}) max_depth={MAXD} iters={args.iters} "
          f"ch={'1' if ch1 else '0'}{'('+args.partial_depth+')' if args.partial_depth else ''} test_rt={args.test_rt or 'canonical'}")

    device = torch.device("cuda:0")
    model = build_model("edge", True, args.load_checkpoints, device, [0])
    if args.dataset == "dsec":
        DatasetCls = DatasetDSEC; data_path = args.data_path or "data/dsec_generated"
    else:
        DatasetCls = DatasetM3ED; data_path = args.data_path or "data/m3ed"
    kw = {"test_RT_path": args.test_rt} if (args.test_rt and args.dataset == "m3ed") else {}
    ds = DatasetCls(data_path, event_representation=ev, max_r=5.0, max_t=0.5,
                    split="test", test_sequence=args.seq, **kw)

    # optional: load saved per-frame poses (our exact reported pose) to use as the refinement init.
    pose_by_i = None
    if args.init_poses is not None:
        mkey = {"median": "conf_gtmediantop50", "all": "all",
                "floor0": "conf_pweightfl0", "floor0.5": "conf_pweightfl0.5"}[args.select]
        npz = np.load(args.init_poses, allow_pickle=True)
        idx_arr = npz[f"{mkey}__i"].astype(int)
        pq = np.asarray(npz[f"{mkey}__pq"]); pt = np.asarray(npz[f"{mkey}__pt"])
        pose_by_i = {int(idx_arr[k]): (pq[k], pt[k]) for k in range(len(idx_arr))}
        print(f"[init_poses] loaded {len(pose_by_i)} '{mkey}' poses from {args.init_poses}")

    frames = args.frames if args.frames is not None else list(range(len(ds)))
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    rows = []
    n_skip = 0
    probe_corr = []
    rows_by_cfg = {c: [] for c in SWEEP_CFGS} if args.sweep_both else None
    for i in frames:
        s = ds[i]
        event_frame = s["event_frame"].unsqueeze(0)
        pc = s["point_cloud"]; calib = s["calib"]
        T_err = s["tr_error"].unsqueeze(0); R_err = s["rot_error"].unsqueeze(0)

        dp = Data_preprocess(calib.unsqueeze(0), 3, 5, partial_fill=args.partial_depth)
        event_input, depth_input, flow_gt, _ = dp.push_fuse(
            event_frame, [pc], T_err, R_err, device,
            MAX_DEPTH=MAXD, split="test", h=CROP_H, w=CROP_W)
        depth_input = depth_input[:, (1 if ch1 else 0), :, :].unsqueeze(1)

        with torch.no_grad():
            flow_predictions, flow_up, depth2edge, conf_raw = model(
                depth_input, event_input, iters=args.iters, test_mode=True, idx=i, output_conf=True)
        conf_map = torch.nn.functional.softplus(conf_raw)[0, 0].cpu().numpy()   # dense [H,W] per-pixel confidence

        # ---- baseline pose ----
        if pose_by_i is not None:
            # our exact reported pose (all / conf>median) as the refinement init.
            if i not in pose_by_i:                 # frame skipped in the pose-eval -> skip here too
                n_skip += 1
                continue
            pq, pt = pose_by_i[i]
            R_pred = torch.tensor(pq).float(); T_pred = torch.tensor(pt).float()
        else:
            R_pred, T_pred, inliers, flag = Flow2Pose(
                flow_up, depth_input, calib.unsqueeze(0), MAX_DEPTH=MAXD,
                x=CROP_X, y=CROP_Y, h=CROP_H, w=CROP_W)
            if flag:
                n_skip += 1
                continue
        base_r, base_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
        base_r = float(base_r); base_t = float(base_t)

        # ---- 3D edge source: model's depth2edge prediction ----
        ep = depth2edge[-1] if isinstance(depth2edge, (list, tuple)) else depth2edge
        ep = ep[0, 0].float()
        if ep.min() < 0 or ep.max() > 1:
            ep = torch.sigmoid(ep)
        ep_np = ep.detach().cpu().numpy()                 # edge-existence probability map (edgeprob confidence source)
        edge_bin = (ep > args.edge_thresh).cpu().numpy().astype(np.float32)
        if args.skeleton_pred:
            edge_bin = skeletonize(edge_bin)
        depth_m = depth_input[0, 0].cpu().numpy() * MAXD

        rc = np.argwhere(edge_bin > 0)
        if len(rc):
            z = depth_m[rc[:, 0], rc[:, 1]]
            keep = z > 0.1
            rc = rc[keep]; z = z[keep]
        if args.edge_subsample > 1 and len(rc) > args.min_pts * args.edge_subsample:
            rc = rc[::args.edge_subsample]; z = z[::args.edge_subsample]   # thin edges (chamfer is over-determined)
        if len(rc) < args.min_pts:
            rows.append((i, base_t, base_r, base_t, base_r, float("nan"), float("nan"), len(rc)))
            continue

        # ---- per-edge-pixel confidence ----
        _epv = ep_np[rc[:, 0], rc[:, 1]].astype(np.float64)             # edge-existence probability
        _cfv = conf_map[rc[:, 0], rc[:, 1]].astype(np.float64)         # flow-correspondence confidence (same head as selection)
        if args.edge_conf_src == "edgeprob":
            conf_edge = _epv
        elif args.edge_conf_src == "both":
            conf_edge = _epv * _cfv                                     # edge is real AND its flow/geometry is reliable
        else:
            conf_edge = _cfv
        weights = None
        if args.edge_conf == "pweight":
            # keep each edge pixel w.p. floor+(1-floor)*conf_norm (per-frame min-max), then uniform-refine the subset
            lo, hi = conf_edge.min(), conf_edge.max()
            cn = (conf_edge - lo) / (hi - lo) if hi > lo else np.ones_like(conf_edge)
            p = args.edge_conf_floor + (1.0 - args.edge_conf_floor) * cn
            ksel = edge_rng.random(len(conf_edge)) < p
            if ksel.sum() >= args.min_pts:                          # guard: don't drop below min_pts
                rc = rc[ksel]; z = z[ksel]; conf_edge = conf_edge[ksel]

        fx, fy, cx, cy = [float(v) for v in calib]
        cxc = cx - CROP_Y                                            # principal point in cropped frame
        cyc = cy - CROP_X
        uv = torch.tensor(np.stack([rc[:, 1], rc[:, 0]], 1), dtype=torch.float32, device=device)
        X = deproject(uv, torch.tensor(z, dtype=torch.float32, device=device), fx, fy, cxc, cyc)
        if args.edge_conf == "weight":
            weights = torch.tensor(conf_edge, dtype=torch.float32, device=device)

        # ---- target: observed event edges ----
        act = event_input[0].abs().sum(0).cpu().numpy()
        occ = (act > 0).astype(np.float32)
        ev_edges = occ if args.no_skeleton_target else skeletonize(occ)
        dt = distance_transform(ev_edges).to(device)

        # ---- init from LEAR pose; sanity-check the conversion ----
        xi_init = lear_to_xi(R_pred, T_pred).to(device).float()
        R_chk, T_chk = xi_to_lear(xi_init)
        chk_r, chk_t = err_Pose(R_chk.cpu(), T_chk.cpu(), R_err[0], T_err[0])
        with torch.no_grad():
            uv0 = project((exp_so3(xi_init[:3]) @ X.T).T + xi_init[3:], fx, fy, cxc, cyc)
            d_init_pts = sample_dt(dt, uv0).cpu().numpy()
            dt_init = float(d_init_pts.mean())
        # PROBE: does edge-pixel confidence predict edge alignment (lower DT)? within-frame Spearman(conf, -DT_init)
        if len(conf_edge) > 20 and np.std(conf_edge) > 0:
            from scipy.stats import spearmanr as _sp
            probe_corr.append(_sp(conf_edge, -d_init_pts).correlation)

        # ---- AMORTIZED sweep: forward done once above; run all 6 both-configs' LM on the cached X/dt/weights ----
        if args.sweep_both:
            for (op, aw) in SWEEP_CFGS:
                xi_o, h = refine_pose(X, dt, (fx, fy, cxc, cyc), init_xi=xi_init,
                                      iters=args.refine_iters, huber_delta=args.huber, z_reg=args.zreg,
                                      anchor_w=float(aw), lock_rot=(op == "transl"), weights=weights)
                Rr, Tr = xi_to_lear(xi_o); rr, rt = err_Pose(Rr.cpu(), Tr.cpu(), R_err[0], T_err[0])
                rows_by_cfg[(op, aw)].append((i, base_t, base_r, float(rt), float(rr), dt_init,
                                              h[-1] if h else float("nan"), len(rc)))
            if i % 100 == 0: print(f"  [{i}/{len(frames)}] swept 6 cfgs  N={len(rc)}", flush=True)
            continue

        # ---- refine ----
        xi_opt, hist = refine_pose(X, dt, (fx, fy, cxc, cyc), init_xi=xi_init,
                                   iters=args.refine_iters, huber_delta=args.huber, z_reg=args.zreg,
                                   anchor_w=args.anchor_w, lock_rot=args.lock_rot, weights=weights)
        R_ref, T_ref = xi_to_lear(xi_opt)
        ref_r, ref_t = err_Pose(R_ref.cpu(), T_ref.cpu(), R_err[0], T_err[0])
        ref_r = float(ref_r); ref_t = float(ref_t)
        dt_ref = hist[-1] if hist else float("nan")

        rows.append((i, base_t, base_r, ref_t, ref_r, dt_init, dt_ref, len(rc)))
        conv_ok = abs(float(chk_t) - base_t) < 1e-2 and abs(float(chk_r) - base_r) < 1e-2
        print(f"{i:04d}: base {base_t:6.2f}cm/{base_r:5.2f}deg -> ref {ref_t:6.2f}cm/{ref_r:5.2f}deg "
              f"| DT {dt_init:5.1f}->{dt_ref:4.1f}px | N={len(rc):4d} | conv={'OK' if conv_ok else 'BAD'}")

    if args.sweep_both:
        for (op, aw), rws in rows_by_cfg.items():
            outp = f"{args.out_prefix}__{args.sweep_src}__{op}__aw{aw}.csv"
            os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
            with open(outp, "w") as f:
                f.write("idx,base_t,base_r,ref_t,ref_r,dt_init,dt_ref,npts\n")
                for r in rws:
                    f.write(",".join(str(x) for x in r) + "\n")
        print(f"\n===== SWEEP_BOTH {args.seq}: {len(rows_by_cfg)} configs -> {args.out_prefix}__{args.sweep_src}__*.csv =====")
        for (op, aw), rws in rows_by_cfg.items():
            if rws:
                rt = np.array([r[3] for r in rws]); print(f"  {args.sweep_src} {op:6} aw{aw:<3}  med_t {np.median(rt):.2f}  (n={len(rws)})")
        return

    # ---- summary ----
    arr = np.array([(r[1], r[2], r[3], r[4]) for r in rows], dtype=np.float64)
    if len(arr):
        bt, br, rt, rr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        print("\n===== SUMMARY %s (n=%d, skipped=%d) =====" % (args.seq, len(arr), n_skip))
        if probe_corr:
            print(f"  PROBE within-frame Spearman(edge_conf, -DT_init) = {np.nanmean(probe_corr):+.3f}  "
                  f"(n={len(probe_corr)} frames; negative-of-DT so + => high conf aligns better)  edge_conf={args.edge_conf}")
        print(f"  BASELINE : mean {bt.mean():.2f}cm / {br.mean():.3f}deg | median {np.median(bt):.2f}cm / {np.median(br):.3f}deg")
        print(f"  REFINED  : mean {rt.mean():.2f}cm / {rr.mean():.3f}deg | median {np.median(rt):.2f}cm / {np.median(rr):.3f}deg")
        imp_t = (rt < bt).mean() * 100; imp_r = (rr < br).mean() * 100
        print(f"  improved : translation {imp_t:.0f}% of frames | rotation {imp_r:.0f}% of frames")
        print(f"  delta    : mean transl {rt.mean()-bt.mean():+.2f}cm | mean rot {rr.mean()-br.mean():+.3f}deg")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("idx,base_t,base_r,ref_t,ref_r,dt_init,dt_ref,npts\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print("  wrote", args.out)


if __name__ == "__main__":
    main()
