"""Extended eval: per selection method report mean+median translation(cm), rotation(deg), acc%,
kept-depth, and EPE(px) OF THE SELECTED points (gre == per-correspondence flow endpoint error under
GT pose). Selections: all / conf top10 / conf >median(top50); + ORACLE top10 as ceiling ref.

--ev_input selects the event representation directory (event_frames_<ev_input>). Defaults to the
falcon/indoor _half rep for backward compatibility; night / fast_flight / DSEC seqs need
--ev_input ours_pre_100000 (see memory: current-training-eval-pipeline)."""
import os, sys, argparse
sys.path.insert(0, os.getcwd()); sys.path.append("core")
import numpy as np, torch, torch.nn.functional as F
from functools import partial
from scipy.stats import spearmanr
import visibility, poselib
from camera_model import CameraModel
from utils_point import quat2mat
from core.datasets_m3ed import DatasetM3ED
from core.datasets_dsec import DatasetDSEC
from core.scene_config import scene_geometry, scene_ev_input
from core.data_preprocess import Data_preprocess
from core.flow2pose import err_Pose
from core.model_builder import build_model
# crop geometry (crop_h, crop_w, crop_x=center-crop height offset, crop_y=width offset), MAX_DEPTH.
# M3ED half-res = 288,512,36,64 ; DSEC (480x640 event frames) = 360,480,60,80 (from main.py preset).
# Set as module globals so get_corr/gt_reproj/solve pick the right values; overridden per --dataset.
CH,CW,CX,CY,MAXD=288,512,36,64,10.; dev=torch.device("cuda:0")
RANSAC={"max_reproj_error":12.0,"seed":0,"progressive_sampling":False,"max_prosac_iterations":50000,"real_focal_check":False}
BUNDLE={'loss_type':"HUBER",'loss_scale':1.0,'gradient_tol':1e-8,'step_tol':1e-8,'initial_lambda':1e-3,'min_lambda':1e-10,'max_lambda':1e10,'verbose':False}

def get_corr(flow_up,depth,calib):
    out=torch.zeros(flow_up.shape).to(dev); pdi=torch.zeros(depth.shape).to(dev)+1000.
    out=visibility.image_warp_index(depth.to(dev),flow_up.int(),pdi,out,depth.shape[3],depth.shape[2])
    pdi[pdi==1000.]=0.; uv=out.cpu().permute(0,2,3,1).numpy()
    di=(depth.cpu().numpy()*MAXD)[0,0]*((uv[0,:,:,0]!=0)+(uv[0,:,:,1]!=0))
    cam=CameraModel(); cp=calib[0].clone().cpu().numpy()
    cp[2]+=CW/2.-(CY+CY+CW)/2.; cp[3]+=CH/2.-(CX+CX+CH)/2.
    cam.focal_length=cp[:2]; cam.principal_point=cp[2:]
    p3,p2,idx=cam.deproject_pytorch(di,uv[0,:,:,:]); return p2,p3,idx,cp
def solve(p2,p3,cp):
    if len(p3)<4: return None
    cam={'model':'SIMPLE_PINHOLE','width':CW,'height':CH,'params':[cp[0],cp[2],cp[3]]}
    try: pose,_=poselib.estimate_absolute_pose(p2.copy(),p3.copy(),cam,RANSAC,BUNDLE)
    except TypeError:
        im,_=poselib.estimate_absolute_pose(p2.copy(),p3.copy(),cam,RANSAC); pose=im.pose
    R=torch.tensor(pose.q); Tt=torch.tensor(pose.t); R[1:]*=-1; Tt*=-1; return R,Tt
def gt_reproj(p3,p2,Re,Te,cp):
    q=np.array([Re[0],-Re[1],-Re[2],-Re[3]]); Rm=quat2mat(torch.tensor(q).float())[:3,:3].numpy().astype(np.float64)
    t=-np.asarray(Te,dtype=np.float64); f,cx,cy=cp[0],cp[2],cp[3]; Xc=(Rm@p3.T).T+t
    proj=np.stack([f*Xc[:,0]/Xc[:,2]+cx,f*Xc[:,1]/Xc[:,2]+cy],1)
    ea=np.linalg.norm(proj-p2,axis=1); eb=np.linalg.norm(proj-p2[:,::-1],axis=1)
    return eb if np.median(eb)<np.median(ea) else ea

def sel_global(conf,frac): n=len(conf); k=max(8,int(round(frac*n))); return np.argsort(-conf)[:k]
def sel_oracle(gre,frac): n=len(gre); k=max(8,int(round(frac*n))); return np.argsort(gre)[:k]
def sel_pweight(conf,floor,rng):
    # probabilistic pseudo-weighting for the weightless poselib solver: keep each correspondence with
    # probability p = floor + (1-floor)*conf_norm, where conf_norm = per-frame min-max of confidence in [0,1].
    # -> lowest-conf point kept w.p. floor, top-conf point w.p. 1; linear in conf between. Soft middle ground
    # between "all" (keep everything) and ">median" (hard top-50% cut); confidence biases the sampled subset.
    c=conf.astype(float); lo=c.min(); hi=c.max()
    cn=(c-lo)/(hi-lo) if hi>lo else np.ones_like(c)
    p=floor+(1.0-floor)*cn
    return np.where(rng.random(len(c))<p)[0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--tag",required=True)
    ap.add_argument("--seq",default="falcon_indoor_flight_3")
    ap.add_argument("--dataset",default="m3ed",choices=["m3ed","dsec"],
                    help="m3ed (DatasetM3ED, crop 288x512) or dsec (DatasetDSEC, crop 360x480, GT=identity)")
    ap.add_argument("--data_path",default=None,help="override data root (default: data/m3ed or data/dsec_generated)")
    ap.add_argument("--ev_input",default=None,help="override event rep dir (default: scene_config per seq)")
    ap.add_argument("--crop",default=None,help="override crop 'CH,CW,CX,CY' (default: scene_config per seq)")
    ap.add_argument("--calib_x2",action="store_true",
                    help="ESCAPE HATCH ONLY: multiply calib by 2. NOT needed — the dataloader now returns the "
                         "correct calib per seq via scene_config (half-res /2, full-res native). Leave off.")
    ap.add_argument("--max_depth",type=float,default=None,help="override max_depth (default: scene_config per seq)")
    ap.add_argument("--dense_depth",action="store_true",help="feed DENSE depth (ch1) instead of sparse (ch0) — must match training")
    ap.add_argument("--partial_depth",default=None,choices=["partial_light"],help="feed the PARTIAL-light completed depth (ch1) instead of sparse — must match training; implies reading ch1. (This is the paper's partial depth completion; 'partial'/'partial_dc' are disabled exploratory variants.)")
    ap.add_argument("--dump",default=None,help="if set, save per-frame (frame_idx,t_err,r_err,depth) per method to this .npz (for section/seed analysis)")
    ap.add_argument("--test_rt",default=None,help="explicit test_RT csv path (perturbation study); default None => canonical auto-named file (unchanged behavior)")
    ap.add_argument("--iters",type=int,default=24,help="iterative flow refinement (IFR) iterations at inference (default 24; train uses 12)")
    ap.add_argument("--save_poses",action="store_true",help="also dump per-frame predicted pose (quaternion+translation, Flow2Pose convention) into the --dump npz, for edge pose-refinement")
    ap.add_argument("--pweight",action="store_true",help="add a probabilistic pseudo-weighted selection: keep each correspondence w.p. p=floor+(1-floor)*conf_norm (per-frame min-max) — approximates confidence weighting inside the weightless poselib solver; soft middle ground between all and >median")
    ap.add_argument("--pweight_floor",type=float,default=0.5,help="floor probability for the lowest-confidence point (default 0.5); top-confidence point kept w.p. 1")
    ap.add_argument("--pweight_floors",default=None,help="comma-separated list of floors to sweep in ONE pass (shares the model forward), e.g. '0,0.3,0.5'; overrides --pweight_floor")
    ap.add_argument("--pweight_seed",type=int,default=0,help="RNG seed for the pweight Bernoulli sampling (reproducibility)")
    ap.add_argument("--pweight_only",action="store_true",help="compute ONLY the pweight method (1 poselib solve/frame) — much faster; use when all/>median already exist from a prior run (correspondences are deterministic at inference, so it's directly comparable)")
    ap.add_argument("--all_only",action="store_true",help="compute ONLY the 'all' method (1 solve/frame) — for baselines (pure LEAR, no confidence)")
    ap.add_argument("--max_r",type=float,default=5.0,help="rotation perturbation margin (deg) for the test set / auto-generated test_RT")
    ap.add_argument("--max_t",type=float,default=0.5,help="translation perturbation margin (m) for the test set / auto-generated test_RT")
    ap.add_argument("--rt_seed",type=int,default=0,help="np.random seed used when GENERATING a missing test_RT (reproducible perturbations)")
    a=ap.parse_args()
    global CH,CW,CX,CY,MAXD
    # scene_config = single source of truth (crop / max_depth / ev_input); flags override.
    g=scene_geometry(a.seq,a.dataset)
    CH,CW,CX,CY = [int(v) for v in a.crop.split(",")] if a.crop else g["crop"]
    MAXD = a.max_depth if a.max_depth is not None else g["max_depth"]
    ev_input = a.ev_input or scene_ev_input(a.seq,a.dataset)
    if a.dataset=="dsec":
        Dataset=DatasetDSEC; data_path=a.data_path or "data/dsec_generated"
    else:
        Dataset=DatasetM3ED; data_path=a.data_path or "data/m3ed"
    print(f"[cfg] seq={a.seq} dataset={a.dataset} class={g['klass']} ev={ev_input} "
          f"crop=({CH},{CW},{CX},{CY}) calib_div={g['calib_div']} calib_x2={a.calib_x2} max_depth={MAXD}",flush=True)
    model=build_model("edge",True,a.ckpt,dev,[0]); K=model.module.conf_head.conv1.weight.shape[1]//128
    _kw={"test_RT_path":a.test_rt} if (a.test_rt and a.dataset=="m3ed") else {}
    np.random.seed(a.rt_seed)   # reproducible perturbations if the test_RT must be generated
    ds=Dataset(data_path,event_representation=ev_input,max_r=a.max_r,max_t=a.max_t,split="test",test_sequence=a.seq,**_kw)
    floors=[float(x) for x in a.pweight_floors.split(",")] if a.pweight_floors else [a.pweight_floor]
    PWMS=[f"conf pweight(fl{f:g})" for f in floors]; PWM=dict(zip(PWMS,floors))
    if a.all_only:
        METHODS=["all"]                                  # baselines: pure LEAR, no confidence selection
    elif a.pweight_only:
        METHODS=list(PWMS)                               # only pweight floor(s) (all/>median reused from prior run)
    else:
        METHODS=["all","conf top10","conf >median(top50)","ORACLE top10"]
        if a.pweight or a.pweight_floors: METHODS+=PWMS
    # per-floor rng (each seeded identically) so a multi-floor pass reproduces each single-floor
    # standalone run bit-for-bit — a shared rng would advance once per floor per frame and diverge.
    rngs={m:np.random.default_rng(a.pweight_seed) for m in PWMS}
    def pick(name,conf,gre):
        if name=="all": return np.arange(len(conf))
        if name=="conf top10": return sel_global(conf,0.1)
        if name=="conf >median(top50)": return sel_global(conf,0.5)
        if name=="ORACLE top10": return sel_oracle(gre,0.1)
        if name in PWM: return sel_pweight(conf,PWM[name],rngs[name])
    T={m:{'t':[],'r':[],'z':[],'epe':[],'i':[],'pq':[],'pt':[]} for m in METHODS}; allz=[]; wf=[]
    for i in range(len(ds)):
        s=ds[i]; ev=s["event_frame"].unsqueeze(0); pc=s["point_cloud"]; calib=s["calib"]; Te=s["tr_error"]; Re=s["rot_error"]
        if a.calib_x2: calib=calib*2.0                              # undo dataloader calib/2 for full-res M3ED
        dp=Data_preprocess(calib.unsqueeze(0),3,5,partial_fill=a.partial_depth)
        ei,di,_,_=dp.push_fuse(ev,[pc],Te.unsqueeze(0),Re.unsqueeze(0),dev,MAX_DEPTH=MAXD,split="test",h=CH,w=CW)
        di=di[:,(1 if (a.dense_depth or a.partial_depth) else 0),:,:].unsqueeze(1)   # ch1=(partial|full) completion, ch0=sparse
        with torch.no_grad(): _,flow_up,_,cf=model(di,ei,iters=a.iters,test_mode=True,output_conf=True,idx=i)
        cm=F.softplus(cf)[0,0].cpu().numpy()
        p2,p3,idx,cp=get_corr(flow_up,di,calib.unsqueeze(0))
        if p3.shape[0]<20: continue
        r=idx[:,0].astype(int); c=idx[:,1].astype(int); conf=cm[r,c]
        gre=gt_reproj(p3.astype(np.float64),p2.astype(np.float64),Re.numpy(),Te.numpy(),cp)
        if len(gre)>20: wf.append(spearmanr(conf,-gre).correlation)
        allz.append(p3[:,2].mean())
        for m in METHODS:
            sel=pick(m,conf,gre)
            if len(sel)<8: continue
            T[m]['epe'].append(float(np.mean(gre[sel])))          # EPE(px) of selected pts (this frame)
            o=solve(p2[sel],p3[sel],cp)
            if o is None: continue
            rr,tt=err_Pose(o[0],o[1],torch.tensor(Re.numpy()),torch.tensor(Te.numpy()))
            T[m]['t'].append(float(tt)); T[m]['r'].append(float(rr)); T[m]['z'].append(p3[sel,2].mean()); T[m]['i'].append(i)
            T[m]['pq'].append(np.asarray(o[0]).ravel()); T[m]['pt'].append(np.asarray(o[1]).ravel())   # predicted pose (quat, transl)
        if i%150==0: print(f"  [{i}/{len(ds)}]",flush=True)
    acc=lambda t,r: 100.0*np.mean((np.array(t)<5.0)&(np.array(r)<5.0))
    acc_out=lambda t,r: 100.0*np.mean((np.array(t)<25.0)&(np.array(r)<2.0))  # outdoor convention
    print(f"\n===== {a.tag} =====")
    print(f"  within-frame Spearman(conf,-err) = {np.nanmean(wf):+.3f}")
    print(f"  median depth ALL = {np.median(allz):.2f}m")
    H=f"  {'method':<22}{'mean_t':>7}{'med_t':>7}{'mean_r':>8}{'med_r':>7}{'acc%':>7}{'accO%':>7}{'depth':>7}{'EPE_mn':>8}{'EPE_md':>8}{'n':>6}"
    print(H); print("  "+"-"*(len(H)-2))
    for m in METHODS:
        t=T[m]['t']
        if not t: print(f"  {m:<22}(none)"); continue
        r_=T[m]['r']; e=T[m]['epe']
        print(f"  {m:<22}{np.mean(t):>7.2f}{np.median(t):>7.2f}{np.mean(r_):>8.3f}{np.median(r_):>7.3f}"
              f"{acc(t,r_):>7.1f}{acc_out(t,r_):>7.1f}{np.median(T[m]['z']):>7.2f}{np.mean(e):>8.2f}{np.median(e):>8.2f}{len(t):>6}")
    if a.dump:
        os.makedirs(os.path.dirname(a.dump) or ".",exist_ok=True)
        safe=lambda s:s.replace(" ","_").replace(">","gt").replace("(","").replace(")","").replace("%","")
        out={}
        for m in METHODS:
            out[f"{safe(m)}__i"]=np.array(T[m]['i'],dtype=np.int32)
            out[f"{safe(m)}__t"]=np.array(T[m]['t'],dtype=np.float32)
            out[f"{safe(m)}__r"]=np.array(T[m]['r'],dtype=np.float32)
            out[f"{safe(m)}__z"]=np.array(T[m]['z'],dtype=np.float32)
            out[f"{safe(m)}__epe"]=np.array(T[m]['epe'],dtype=np.float32)   # per-frame EPE(px) of selected pts -> mean/median downstream
            if a.save_poses and T[m]['pq']:
                out[f"{safe(m)}__pq"]=np.array(T[m]['pq'],dtype=np.float32)   # (n,4) predicted quaternion
                out[f"{safe(m)}__pt"]=np.array(T[m]['pt'],dtype=np.float32)   # (n,3) predicted translation
        np.savez(a.dump,ntest=len(ds),iters=a.iters,**out)
        print(f"  [dump] per-frame errors -> {a.dump}")

if __name__=="__main__":
    main()
