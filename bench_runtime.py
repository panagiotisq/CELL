#!/usr/bin/env python3
"""Runtime + memory benchmark, LEAR Table-style: feature encoding vs. flow estimation (N iters),
plus peak GPU memory. Times the SAME backbone with the confidence head on (ours) and off (baseline),
and at 24 vs 12 IFR iterations. Uses one real frame from an M3ED sequence at the eval resolution.

Usage:
  python bench_runtime.py --ckpt checkpoints/study/falcon_indoor__decpl_plight.pth \
      --seq falcon_indoor_flight_3 --partial_depth --reps 100
"""
import os, sys, argparse, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.datasets_m3ed import DatasetM3ED
from core.datasets_dsec import DatasetDSEC
from core.scene_config import scene_geometry, scene_ev_input
from core.data_preprocess import Data_preprocess
from core.model_builder import build_model
from tools.pose_refine.edge_extract import skeletonize
from tools.pose_refine.refine import distance_transform, deproject, refine_pose


def ev_ms(a, b):
    return a.elapsed_time(b)  # ms between two recorded CUDA events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", default="falcon_indoor_flight_3")
    ap.add_argument("--dataset", default="m3ed", choices=["m3ed", "dsec"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--partial_depth", action="store_true", help="feed the partial-light completion (ch1)")
    ap.add_argument("--dense_depth", action="store_true")
    ap.add_argument("--frame", type=int, default=1, help="frame index to benchmark on")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters_list", default="24,12")
    ap.add_argument("--refine_op", default="full", choices=["full", "transl"])
    ap.add_argument("--refine_aw", type=float, default=30.0)
    a = ap.parse_args()

    dev = "cuda"
    g = scene_geometry(a.seq, a.dataset)
    CH, CW, CX, CY = g["crop"]; MAXD = g["max_depth"]
    ev_input = scene_ev_input(a.seq, a.dataset)
    Dataset = DatasetM3ED if a.dataset == "m3ed" else DatasetDSEC
    data_path = a.data_path or ("data/m3ed" if a.dataset == "m3ed" else "data/dsec_generated")

    model = build_model("edge", True, a.ckpt, dev, [0])
    net = model.module
    ds = Dataset(data_path, event_representation=ev_input, max_r=5.0, max_t=0.5,
                 split="test", test_sequence=a.seq)

    # build one real input pair exactly like eval_full.py
    s = ds[a.frame]
    ev = s["event_frame"].unsqueeze(0); pc = s["point_cloud"]; calib = s["calib"]
    Te = s["tr_error"]; Re = s["rot_error"]
    dp = Data_preprocess(calib.unsqueeze(0), 3, 5, partial_fill=a.partial_depth)
    ei, di, _, _ = dp.push_fuse(ev, [pc], Te.unsqueeze(0), Re.unsqueeze(0), dev,
                                MAX_DEPTH=MAXD, split="test", h=CH, w=CW)
    di = di[:, (1 if (a.dense_depth or a.partial_depth) else 0), :, :].unsqueeze(1)
    print(f"[cfg] seq={a.seq} res={tuple(di.shape[-2:])} partial={a.partial_depth} reps={a.reps}", flush=True)

    iters_list = [int(x) for x in a.iters_list.split(",")]

    def mem_probe(iters, output_conf):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            net(di, ei, iters=iters, test_mode=True, output_edge=True, output_conf=output_conf)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() / (1024**2),
                torch.cuda.max_memory_reserved() / (1024**2))

    def run(iters, output_conf):
        net._bench = True
        enc = flow = conf = 0.0
        for _ in range(a.warmup):
            with torch.no_grad():
                net(di, ei, iters=iters, test_mode=True, output_edge=True, output_conf=output_conf)
        torch.cuda.synchronize()
        for _ in range(a.reps):
            with torch.no_grad():
                net(di, ei, iters=iters, test_mode=True, output_edge=True, output_conf=output_conf)
            torch.cuda.synchronize()
            bev = net._bev
            enc += ev_ms(bev['enc0'], bev['enc1'])
            flow += ev_ms(bev['enc1'], bev['flow1'])
            if output_conf:
                conf += ev_ms(bev['flow1'], bev['conf1'])
        net._bench = False
        mem_a, mem_r = mem_probe(iters, output_conf)
        n = a.reps
        return dict(enc=enc/n, flow=flow/n, conf=conf/n, mem_a=mem_a, mem_r=mem_r)

    # ---- pose-refinement stage timing (ours only): edge-extract + DT + LBFGS ----
    def build_refine_inputs(iters):
        with torch.no_grad():
            _, flow_up, d2e, cf = net(di, ei, iters=iters, test_mode=True, output_edge=True, output_conf=True)
        ep = d2e[-1][0, 0].float()
        if ep.min() < 0 or ep.max() > 1: ep = torch.sigmoid(ep)
        edge_bin = ep.cpu().numpy() > 0.5
        depth_m = di[0, 0].cpu().numpy() * float(MAXD)
        rc = np.argwhere(edge_bin)
        z = depth_m[rc[:, 0], rc[:, 1]]; keep = z > 0.1; rc = rc[keep]; z = z[keep]
        fx, fy, cx, cy = [float(v) for v in calib]
        cxc = cx + CW / 2. - (CY + CY + CW) / 2.; cyc = cy + CH / 2. - (CX + CX + CH) / 2.
        uv = torch.tensor(np.stack([rc[:, 1], rc[:, 0]], 1), dtype=torch.float32, device=dev)
        X = deproject(uv, torch.tensor(z, dtype=torch.float32, device=dev), fx, fy, cxc, cyc)
        act = ei[0].abs().sum(0).cpu().numpy()
        cam_crop = (fx, fy, cxc, cyc)
        return X, act, cam_crop, X.shape[0]

    def time_refine(iters, op="full", anchor_w=30.0, refine_iters=80):
        X, act, cam_crop, npts = build_refine_inputs(iters)
        xi0 = torch.zeros(6, device=dev)
        lock = (op == "transl")
        pre = lbfgs = 0.0
        for r in range(a.warmup + a.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            tgt = skeletonize((act > 0).astype(np.float32)); dt = distance_transform(tgt).to(dev)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            refine_pose(X, dt, cam_crop, init_xi=xi0, iters=refine_iters, huber_delta=3.0,
                        z_reg=1e-3, anchor_w=anchor_w, lock_rot=lock)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if r >= a.warmup:
                pre += (t1 - t0) * 1000; lbfgs += (t2 - t1) * 1000
        return dict(pre=pre / a.reps, lbfgs=lbfgs / a.reps, npts=npts)

    print("\n%-24s %9s %9s %9s %10s %10s" % ("config", "encode", "flow", "conf", "alloc MB", "reserv MB"))
    print("-" * 78)
    r = run(iters_list[0], output_conf=False)
    print("%-24s %9.2f %9.2f %9s %10.0f %10.0f" % (f"baseline @{iters_list[0]}", r['enc'], r['flow'], "--", r['mem_a'], r['mem_r']))
    for it in iters_list:
        r = run(it, output_conf=True)
        print("%-24s %9.2f %9.2f %9.2f %10.0f %10.0f" % (f"ours @{it}", r['enc'], r['flow'], r['conf'], r['mem_a'], r['mem_r']))

    print("\n%-24s %12s %12s %10s" % ("refinement stage", "edge+DT ms", "LBFGS ms", "#edge pts"))
    print("-" * 62)
    rf = time_refine(iters_list[0], op=a.refine_op, anchor_w=a.refine_aw)
    print("%-24s %12.2f %12.2f %10d" % (f"ours refine ({a.refine_op})", rf['pre'], rf['lbfgs'], rf['npts']))

    print("\nnote: encode=fnet+corr+cnet (once); flow=IFR loop; conf=confidence head (once);")
    print("      refinement = edge-skeletonize+distance-transform (CPU) + LBFGS se(3) (GPU).")


if __name__ == "__main__":
    main()
