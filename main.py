import os
import sys
import time 
import csv

import cv2
import numpy as np
import argparse
import random
import torch
import torch.nn.functional as F

from core.datasets_m3ed import DatasetM3ED
from core.datasets_dsec import DatasetDSEC
# from core.datasets_mvsec import DatasetMVSEC
from core.backbone import Backbone_Event, Backbone_Edge, Backbone_Edge_FF
from core.utils import (count_parameters, merge_inputs, fetch_optimizer, Logger)
from core.utils_point import overlay_imgs
from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose, err_Pose
from core.scene_config import scene_geometry
from core.losses import warp, sequence_loss, sequence_loss_attn, sequence_loss_reweight, sequence_loss_zero, correlation_matching_loss, build_flow_target, ClassifyLoss, ProposedLoss
from core.pose_e2e import PoseE2ELoss
from core.flow_viz import flow_to_image

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer): 
            optimizer.step()

        def update(self):
            pass
    

def _init_fn(worker_id, seed):
    seed = worker_id + seed
    print(f"Init worker {worker_id} with seed {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch, occlusion_kernel=5, occlusion_threshold=3, pose_e2e_loss=None):
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel, partial_fill=args.partial_depth)
        # crop from scene_config (per-sequence resolution): half 288x512 / full-res night 600x960 / DSEC 360x480
        crop_h, crop_w, crop_x, crop_y = args._scene_crop
        # # M3ED Full Resolution
        # crop_h, crop_w, crop_x, crop_y = 480, 960, 120, 160
        # # night, fast
        # crop_h, crop_w, crop_x, crop_y = 600, 960, 60, 160
        # # MVSEC
        # crop_h, crop_w, crop_x, crop_y = 240, 320, 10, 13
        # # DSEC
        # crop_h, crop_w, crop_x, crop_y = 360, 480, 60, 80
        if args.backbone == "baseline":
            event_input, depth_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=crop_h, w=crop_w)
        elif args.backbone == "edge":
            edge_masks = sample['edge_mask'] if args.edge_gt else None
            # --dense_flow: supervise the flow loss on the completed partial-depth support — exact GT flow
            # on real points + a geometrically-consistent target on the newly-filled partial_light pixels;
            # encoder input and edge-GT are unchanged vs plain partial_light.
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=crop_h, w=crop_w, edge_masks=edge_masks, dense_flow=args.dense_flow)
            depth_input_full = depth_input                     # [B,3,H,W]: ch0 perturbed-sparse, ch1 perturbed-dense, ch2 GT-pose-sparse (=D_GT, for --zero_flow)
            depth_input = depth_input[:, (1 if (args.dense_depth or args.partial_depth) else 0), :, :].unsqueeze(1)  # ch1=(partial|full) dense, ch0=sparse
        else:
            raise "Specified backbone doesn't exist"

        # visualization_folder = f"./visualization/{args.backbone}/train"
        # if not os.path.exists(visualization_folder):
        #     os.makedirs(f"{visualization_folder}/train")
        #     os.makedirs(f"{visualization_folder}/test")
        # vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        # vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        # vis_event_time_image = vis_event_time_image[:, :, [2, 0, 1]]
        # cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_1_1_event_input.png", (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8))
        # vis_depth_input = overlay_imgs(event_input[0, :3, :, :]*0, depth_input[0, 0, :, :])
        # cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_2_1_depth_input.png", (vis_depth_input / np.max(vis_depth_input) * 255).astype(np.uint8))
        # flow_viz = flow_to_image(flow_gt[0, ...].permute(1,2,0).cpu().detach().numpy())
        # cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_1_flow_gt.png", flow_viz) 
        if args.backbone == "edge":
            ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
        #     cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))

        if args.bin:
            event_input[event_input > 0] = 1.

        optimizer.zero_grad()
        if args.backbone == "baseline":
            flow_preds = model(depth_input, event_input, iters=args.train_iters)
            loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
            flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
            cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
        elif args.backbone == "edge":
            if args.conf_head:
                # conf_detach controls whether the conf head's grad reaches the shared flow backbone.
                #   JOINT pose_e2e   -> False (flow+conf couple; pose grad also trains flow via x2d)
                #   DECOUPLED pose_e2e-> True  (conf trains on pose loss alone; flow untouched by conf)
                #   coadapt          -> False;  detached conf loss -> True
                if args.pose_e2e:
                    conf_detach = args.pose_e2e_decouple
                else:
                    conf_detach = not args.conf_attenuate
                flow_preds, depth2edge_preds, conf_pred = model(depth_input, event_input,
                    iters=args.train_iters, output_conf=True, conf_detach=conf_detach)
            else:
                flow_preds, depth2edge_preds = model(depth_input, event_input, iters=args.train_iters)
            ## flow loss: attenuated (coadapt) | conf-reweighted (pose_e2e decoupled) | plain (joint / default)
            if args.conf_head and args.conf_attenuate and not args.pose_e2e:
                c = F.softplus(conf_pred) + 1e-4
                loss_flow, metrics = sequence_loss_attn(flow_preds, flow_gt, c,
                    conf_alpha=args.conf_alpha, conf_scale=args.conf_scale, gamma=args.gamma)
                metrics['conf_mean'] = metrics.get('conf_mean', 0.0)
            elif args.pose_e2e and args.pose_e2e_decouple:
                # idea-2: flow focuses on the (detached) pose-supervised high-conf regions
                w_pix = F.softplus(conf_pred).detach()
                if args.pose_e2e_reweight_temp != 1.0:
                    # Sweep B: temperature on the flow-reweighting. >1 SHARPENS focus onto high-conf
                    # pixels (conf steers the flow harder); <1 FLATTENS toward uniform. Only relative
                    # weights matter (sequence_loss_reweight takes a weighted mean). softplus>0 -> no zeros.
                    w_pix = w_pix ** args.pose_e2e_reweight_temp
                loss_flow, metrics = sequence_loss_reweight(flow_preds, flow_gt, w_pix, args.gamma, MAX_FLOW=400)
            else:
                loss_flow, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
            # flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
            # cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
            loss_edge, loss_edge_last = ClassifyLoss(depth2edge_preds, ground_truth_depth2edge, loss_func="Sequence_Weighted_Cross_Entropy_Loss")
            metrics['edge_loss'] = loss_edge_last.item()
            # cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge_preds[-1][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            alpha = 1
            beta = 100
            loss = alpha * loss_flow + beta * loss_edge
            if args.conf_head and not args.conf_attenuate and not args.pose_e2e:
                # DETACHED confidence loss (EGFS upper branch, pixel r_hat, valid = GT-flow available).
                # conf_pred came from net.detach() -> this loss only trains the conf head.
                with torch.no_grad():
                    f_fin = flow_preds[-1]                                            # [B,2,H,W] final flow
                    err = torch.norm(f_fin - flow_gt, dim=1, keepdim=True)            # pixel flow error
                    valid = ((flow_gt[:, 0:1] != 0) | (flow_gt[:, 1:2] != 0)).float() # ~6% GT-flow mask
                    if args.conf_no_tanh:
                        # linear r_hat, robust clamp -> preserves full-range ranking (no tail saturation)
                        r_hat = torch.clamp(err / args.conf_scale, max=args.conf_clamp)
                    else:
                        r_hat = torch.tanh(err / args.conf_scale)                     # tanh-clamped
                c = F.softplus(conf_pred) + 1e-4                                      # positive confidence
                conf_terms = c * r_hat - args.conf_alpha * torch.log(c)              # ci*r_hat - a*log ci
                loss_conf = (valid * conf_terms).sum() / valid.sum().clamp(min=1.0)
                metrics['conf_loss'] = loss_conf.item()
                loss = loss + args.conf_weight * loss_conf
            if args.pose_e2e:
                # STAGE 5: EPro-PnP Monte-Carlo pose loss over differentiable flow correspondences.
                # Grad flows into flow_preds[-1] (via x2d) AND conf_pred (via w2d). Plain flow loss
                # above anchors the flow; this term makes it pose-relevant & depth-unbiased.
                # cast to fp32: the MC loss (cholesky / sampling) is numerically unsafe under autocast fp16
                # decoupled -> detach_flow (pose loss trains only the conf head)
                pose_loss, pm = pose_e2e_loss(
                    flow_preds[-1].float(), conf_pred.float(), depth_input.float(), calib, T_err, R_err,
                    detach_flow=args.pose_e2e_decouple)
                warm = min(1.0, (logger.total_steps + 1) / max(1, args.pose_e2e_warmup))
                loss = loss + args.pose_e2e_weight * warm * pose_loss
                metrics.update(pm); metrics['pose_lambda'] = args.pose_e2e_weight * warm
            if args.zero_flow and not args.zero_flow_encoder_only:
                # ZERO-FLOW BRANCH, FULL-NETWORK (1a): 2nd forward, SAME shared weights, on the GT-pose
                # (aligned) depth ch2 -> true displacement is 0 wherever the depth projects. Supervise
                # flow -> 0 (masked EPE, sequence_loss_zero) => pushes the correlation to peak at
                # displacement 0 => co-located depth/event features must match. Discarded at inference.
                # (encoder-only variant 2a is handled AFTER the main backward, see below.)
                depth_gt_in = depth_input_full[:, 2, :, :].unsqueeze(1)          # D_GT (aligned)
                mask_zero = depth_input_full[:, 2, :, :] > 0                      # (B,H,W) aligned-depth valid (~15-20%)
                if args.zero_flow_conf:
                    # 1b: conf-WEIGHTED zero loss -- reweight EXACTLY like the decoupled main loss:
                    # w = softplus(conf).detach() (conf from the zero pass, conf_detach so it doesn't
                    # touch the flow backbone), same reweight_temp. Focuses aligned-matching on the
                    # pose-relevant (high-conf) pixels. epe_zero stays unweighted (comparable to 1a).
                    fz_preds, _, fz_conf = model(depth_gt_in, event_input, iters=args.train_iters,
                                                 output_conf=True, conf_detach=True)
                    w_pix = F.softplus(fz_conf).detach()
                    if args.pose_e2e_reweight_temp != 1.0:
                        w_pix = w_pix ** args.pose_e2e_reweight_temp
                    loss_zero, zm = sequence_loss_zero(fz_preds, mask_zero, args.gamma, conf_w=w_pix)
                else:
                    fz_preds, _ = model(depth_gt_in, event_input, iters=args.train_iters)
                    loss_zero, zm = sequence_loss_zero(fz_preds, mask_zero, args.gamma)
                loss = loss + args.zero_flow_weight * loss_zero
                # instrumentation: zero_loss (signal#1 = is the branch learnable), epe_zero (signal#2 =
                # aligned-input flow -> 0 = correlation peak sharpening). main 'epe' already logged (signal#3).
                metrics.update(zm); metrics['zero_w'] = args.zero_flow_weight
                metrics['zero_loss'] = loss_zero.item()
            if args.corr_match:
                # OPTION 3 (correlation-level matching, GRU-free): encoders-only 2nd forward on the
                # GT-pose (aligned) depth ch2 + events -> build the full correlation and push its
                # identity diagonal to be the row-max (softmax CE). Direct one-hop signal to both
                # fnet_lidar (F_D) and fnet_event (F_EV); NO GRU in the path (cheaper + cleaner than
                # the zero-flow branch). 0 extra params; discarded at inference. Opt-in, additive.
                if args.corr_match_flow:
                    # VERSION B: supervise the REAL (perturbed-input) correlation to peak at the
                    # GT-flow-shifted target q = p + flow_gt(p). depth_input here is ch0 (perturbed).
                    fm1, fm2 = model.module.corr_match_features(depth_input, event_input)
                    fv_full = (depth_input_full[:, 0:1, :, :] > 0)               # where flow_gt is defined
                    rows_m = F.max_pool2d(fv_full.float(), 8, 8)[:, 0] > 0       # 1/8 source cells to supervise
                    ve = F.max_pool2d((event_input.abs().sum(1, keepdim=True) > 0).float(), 8, 8)[:, 0] > 0
                    tgt_idx = build_flow_target(flow_gt, fv_full)                # (B,h,w) long flow-shifted target
                else:
                    # VERSION A: aligned depth (ch2), identity target (q = p).
                    depth_gt_in = depth_input_full[:, 2, :, :].unsqueeze(1)      # D_GT (aligned)
                    fm1, fm2 = model.module.corr_match_features(depth_gt_in, event_input)
                    rows_m = F.max_pool2d((depth_input_full[:, 2:3, :, :] > 0).float(), 8, 8)[:, 0] > 0
                    ve = F.max_pool2d((event_input.abs().sum(1, keepdim=True) > 0).float(), 8, 8)[:, 0] > 0
                    rows_m = rows_m & ve                                         # identity needs source event-active
                    tgt_idx = None
                # conf-weighted variant: weight each row's CE by the DETACHED softplus(conf) from the
                # MAIN forward (same reweight as decoupled), downsampled to 1/8 -> matching pressure
                # focuses on pose-relevant high-conf pixels. Needs the conf head (conf_pred in scope).
                row_w = None
                if args.corr_match_conf and args.conf_head:
                    w_full = F.softplus(conf_pred).detach()                      # (B,1,H,W)
                    if args.pose_e2e_reweight_temp != 1.0:
                        w_full = w_full ** args.pose_e2e_reweight_temp
                    row_w = F.avg_pool2d(w_full, 8, 8)[:, 0]                     # (B,h,w) mean conf per 1/8 cell
                loss_cm, cmm = correlation_matching_loss(fm1, fm2, rows_m, ve,
                                                         normalize=args.corr_match_norm,
                                                         temperature=args.corr_match_temp,
                                                         row_weight=row_w,
                                                         target_index=tgt_idx,
                                                         soft_sigma=args.corr_match_soft)
                loss = loss + args.corr_match_weight * loss_cm
                metrics.update(cmm); metrics['cm_w'] = args.corr_match_weight
                metrics['cm_conf'] = 1.0 if row_w is not None else 0.0
                metrics['cm_flow'] = 1.0 if args.corr_match_flow else 0.0
        else:
            raise "Specified backbone doesn't exist"

        scaler.scale(loss).backward()
        if args.zero_flow and args.zero_flow_encoder_only:
            # ZERO-FLOW ENCODER-ONLY (2a): the zero-flow gradient trains ONLY the feature encoders that
            # build the correlation -- fnet_lidar (F_D/F_ED) + fnet_event (F_EV) -- NOT the update block,
            # context encoder, edge branch, or conf head. Mechanism: (i) requires_grad=False on all non-fnet
            # params -> their WEIGHTS don't update, but gradient still flows THROUGH them (the frozen GRU/
            # context) to the encoders via the correlation; (ii) non-fnet modules set to eval() so the
            # aligned-depth pass does NOT update their BatchNorm running stats (fnet is InstanceNorm, so
            # unaffected). Separate backward accumulates onto the encoders' main-branch grads. Same uniform
            # zero-loss + weight as 1a -> the ONLY variable vs 1a is the gradient routing.
            saved_rg = [(p, p.requires_grad) for p in model.parameters()]
            for _n, _p in model.named_parameters():
                _p.requires_grad_(('fnet_lidar' in _n) or ('fnet_event' in _n))
            model.eval(); model.module.fnet_lidar.train(); model.module.fnet_event.train()
            depth_gt_in = depth_input_full[:, 2, :, :].unsqueeze(1)          # D_GT (aligned)
            fz_preds, _ = model(depth_gt_in, event_input, iters=args.train_iters)
            mask_zero = depth_input_full[:, 2, :, :] > 0                     # (B,H,W) aligned-depth valid
            loss_zero, zm = sequence_loss_zero(fz_preds, mask_zero, args.gamma)
            scaler.scale(args.zero_flow_weight * loss_zero).backward()
            for _p, _rg in saved_rg:
                _p.requires_grad_(_rg)
            model.train()
            metrics.update(zm); metrics['zero_w'] = args.zero_flow_weight
            metrics['zero_loss'] = loss_zero.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        logger.push(metrics)


def test(args, TestImgLoader, model, device, occlusion_kernel=5, occlusion_threshold=3, is_test=False):
    model.eval()

    out_list, epe_list = [], []
    Time = 0.
    outliers, err_r_list, err_t_list = [], [], []
    pose_loss = []
    inlier_rate = 0.

    epe_iter_list = []
    for _ in range(args.iteration_num):
        epe_iter_list.append(0)

    pose_loss_fn = ProposedLoss(1., 1.)

    if args.save_log:
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        log_file = f'./logs/{args.backbone}_M3ED_{args.test_sequence}.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
        log_file.writerow(header)
    
    for i_batch, sample in enumerate(TestImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel, partial_fill=args.partial_depth)
        # crop from scene_config (per-sequence resolution): half 288x512 / full-res night 600x960 / DSEC 360x480
        crop_h, crop_w, crop_x, crop_y = args._scene_crop
        # # M3ED Full Resolution
        # crop_h, crop_w, crop_x, crop_y = 480, 960, 120, 160
        # # night, fast
        # crop_h, crop_w, crop_x, crop_y = 600, 960, 60, 160
        # # MVSEC
        # crop_h, crop_w, crop_x, crop_y = 240, 320, 10, 13
        # # DSEC
        # crop_h, crop_w, crop_x, crop_y = 360, 480, 60, 80
        if args.backbone == "baseline":
            event_input, depth_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=crop_h, w=crop_w)

        elif args.backbone == "edge":
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=crop_h, w=crop_w)
            depth_input = depth_input[:, (1 if (args.dense_depth or args.partial_depth) else 0), :, :].unsqueeze(1)  # ch1=(partial|full) dense, ch0=sparse
        else:
            raise "Specified backbone doesn't exist"

        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
        val = valid_gt.view(-1) >= 0.5

        if args.bin:
            event_input[event_input > 0] = 1.

        end = time.time()
        if args.backbone == "baseline":
            flow_predictions, flow_up = model(depth_input, event_input, iters=args.test_iters, test_mode=True, idx=i_batch)
        elif args.backbone == "edge":
            flow_predictions, flow_up, depth2edge = model(depth_input, event_input, iters=args.test_iters, test_mode=True, idx=i_batch)
            # flow_predictions, flow_up = model(depth_input, event_input, iters=args.test_iters, test_mode=True, idx=i_batch, output_edge=False)
        Time += time.time() - end

        # visualization_folder = f"./visualization/{args.backbone}"
        # if not os.path.exists(visualization_folder):
        #     os.makedirs(f"{visualization_folder}/train")
        #     os.makedirs(f"{visualization_folder}/test")
        # vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        # vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        # vis_event_time_image = vis_event_time_image[:, :, [2, 0, 1]]
        # vis_event_time_image = (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8)
        # invalid = (vis_event_time_image[:, :, 0] + vis_event_time_image[:, :, 1] + vis_event_time_image[:, :, 2]) == 0
        # vis_event_time_image[invalid] += 255
        # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_1_1_event_input.png", vis_event_time_image)
        # vis_depth_input = overlay_imgs(event_input[0, :3, :, :]*0, depth_input[0, 0, :, :])
        # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_2_1_depth_input.png", (vis_depth_input / np.max(vis_depth_input) * 255).astype(np.uint8))
        # flow_viz = flow_to_image(flow_gt[0, ...].permute(1,2,0).cpu().detach().numpy())
        # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_1_flow_gt.png", flow_viz) 
        # flow_viz = flow_to_image(flow_up[0, ...].permute(1,2,0).cpu().detach().numpy())
        # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
        # warp_vis_event_time_image = warp(event_input, flow_up)
        # warp_vis_event_time_image = warp_vis_event_time_image[0,...].permute(1, 2, 0).cpu().detach().numpy()
        # warp_vis_event_time_image = np.concatenate((np.zeros([warp_vis_event_time_image.shape[0], warp_vis_event_time_image.shape[1], 1]), warp_vis_event_time_image), axis=2)
        # warp_vis_event_time_image = warp_vis_event_time_image[:, :, [2, 0, 1]]
        # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_4_1_warp_event_input.png", (warp_vis_event_time_image / np.max(warp_vis_event_time_image) * 255).astype(np.uint8))
        # if args.backbone == "edge":
        #     ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
        #     cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
        #     if args.use_feature_fusion:
        #         cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{args.iteration_num}.png', (depth2edge[-1][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
        #         # for i in range(len(depth2edge)):
        #         #     cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{i:02d}.png', (depth2edge[i][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
        #         #     flow_viz = flow_to_image(flow_predictions[i][0, ...].permute(1,2,0).cpu().detach().numpy())
        #         #     cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_2_flow_pred_{i:02d}.png", flow_viz)
        #     else:
        #         cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{args.iteration_num}.png', (depth2edge[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))

        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        if np.isnan(epe[val].mean().item()):
            outliers.append(i_batch)
            continue
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # end = time.time()
        R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, depth_input, calib, MAX_DEPTH=args.max_depth, x=crop_x, y=crop_y, h=crop_h, w=crop_w)
        # Time += time.time() - end
        inlier_rate += len(inliers) / (depth_input > 0).sum().cpu().detach().numpy()

        # # visualize inliers
        # mask = np.zeros((depth_input.shape[2], depth_input.shape[3]), dtype=np.uint8)
        # for y, x in inliers:
        #     if 0 <= y < depth_input.shape[2] and 0 <= x < depth_input.shape[3]:
        #         mask[y, x] = 1
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(6, 6))
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f"./visualization/inlier/{args.backbone}/{i_batch:05d}.png")

        if args.save_log:
            predicted_T = T_pred.cpu().numpy()
            predicted_R = R_pred.cpu().numpy()
            log_string = [f"{i_batch}", str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0])]
            log_file.writerow(log_string)

        if flag:
            outliers.append(i_batch)
            continue
        else:
            pose_loss_i = pose_loss_fn(T_err, R_err, T_pred.unsqueeze(0), R_pred.unsqueeze(0))
            pose_loss.append(pose_loss_i.item())
            if is_test:
                err_r, err_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())
                print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} | {np.median(err_t_list):.5f} "
                        f"{np.median(err_r_list):.5f} | {np.mean(pose_loss):.5f} | {np.mean(np.array(epe_list)):.5f} | "
                        f"{inlier_rate / (i_batch+1):.5f} | {len(outliers)} | {Time / (i_batch+1):.5f}")
                # # Define text properties
                # org = (flow_viz.shape[1]-400, flow_viz.shape[0]-70)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale = 1
                # color = (0, 0, 255)
                # thickness = 2
                # text = f"R={err_r.item():.3f} T={err_t.item():.3f}"
                # cv2.putText(flow_viz, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
              
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.median(epe_list)
    # epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    pose_loss = np.mean(pose_loss)

    if not is_test:
        return epe, f1, pose_loss
    else:
        return err_t_list, err_r_list, outliers, Time, epe, f1, pose_loss, inlier_rate   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        metavar='DIR',
                        default='data/m3ed',
                        help='path to dataset')
    parser.add_argument('--ev_input',
                        '--event_representation',
                        type=str,
                        default='ours_denoise_stc_trail_pre_100000_half')
    parser.add_argument('--dataset', type=str, default='m3ed', choices=['m3ed', 'dsec'],
                        help="m3ed (DatasetM3ED, per-scene) or dsec (DatasetDSEC, single model on all train seqs).")
    parser.add_argument('--dense_depth', action='store_true',
                        help="Feed the DENSE completed depth (ch1) to the encoder instead of sparse (ch0). "
                             "ch1 is already computed in push_fuse (currently discarded) -> zero extra compute. "
                             "Must match at train + eval. See UniCalib (depth completion = dominant lever).")
    parser.add_argument('--partial_depth', default=None, choices=['partial_light'],  # paper's partial completion; 'partial'/'partial_dc' disabled (exploratory)
                        help="EXPERIMENTAL (opt-in): build ch1 with the given PARTIAL completion and feed it "
                             "(partial=small-gap fill; partial_light=less; partial_dc=+depth-consistent edge gate). "
                             "Unset => sparse/full-dense unchanged. Must match at train + eval.")
    parser.add_argument('--dense_flow', action='store_true',
                        help="Supervise the flow loss on the completed partial-depth support (not only the "
                             "sparse ~10%% GT-flow points): back-project the partial_light completed depth to 3D, "
                             "warp by the GT pose, and recompute a geometrically-consistent flow target over that "
                             "support. The encoder input (ch1 partial_light) is unchanged. Requires --backbone edge; "
                             "pair with --partial_depth partial_light.")
    parser.add_argument('--dual_cam', action='store_true',
                        help="For the single-sequence temporal-split scenes (spot_*), also train on the "
                             "RIGHT-camera frames (~2x data for these data-starved scenes). __getitem__ "
                             "applies the right calib + stereo extrinsic. TEST stays left-only (eval unchanged).")
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help="Override checkpoint dir (fixed, shared across resume windows). Default: auto datetime dir.")
    parser.add_argument('--stop_epoch', type=int, default=None,
                        help="Stop THIS run after this epoch index (OneCycleLR still spans --epochs). For resume chaining.")
    parser.add_argument('--test_sequence',
                        type=str,
                        default='falcon_indoor_flight_3')
    # SHARP EDGE GT (depth ∩ edge-events instead of depth ∩ all-events). Off by default.
    parser.add_argument('--edge_gt', action='store_true',
                        help="Restrict the depth2edge GT to event EDGES (extract_edges on the "
                             "event frame) instead of all events. Input frame unchanged.")
    parser.add_argument('--edge_gt_patch', type=int, default=3)
    parser.add_argument('--edge_gt_tau', type=float, default=80)
    parser.add_argument('--edge_gt_dilate', type=int, default=1)
    parser.add_argument('--run_tag', type=str, default='',
                        help="Label appended to the checkpoint dir to distinguish runs.")
    # ---- DETACHED confidence head (Plan A): trains alongside LEAR, gradients detached ----
    parser.add_argument('--conf_head', action='store_true',
                        help="Train a detached per-pixel confidence head on the flow-residual "
                             "hidden state `net` (EGFS upper-branch loss, pixel r_hat).")
    parser.add_argument('--conf_alpha', type=float, default=10.0,
                        help="Confidence regularizer weight (alpha in ci*r_hat - alpha*log ci).")
    parser.add_argument('--conf_scale', type=float, default=5.0,
                        help="tanh clamp scale for the pixel error r_hat = tanh(err/scale).")
    parser.add_argument('--conf_weight', type=float, default=1.0,
                        help="Weight of the detached confidence loss in the total loss.")
    parser.add_argument('--conf_kernel', type=int, default=3, choices=[1, 3],
                        help="Conf head conv kernel: 3 = LEAR-style local mixing, 1 = pure "
                             "per-correspondence readout of the local net vector.")
    parser.add_argument('--conf_feats', type=str, default='net',
                        help="Comma list of features feeding the conf head: {net,cnet,motion}. "
                             "'net'=K GRU-state snapshots; 'cnet'=upstream context (freer to encode "
                             "confidence under co-adaptation); 'motion'=encoded corr+flow (final iter).")
    parser.add_argument('--conf_nets', type=int, default=1,
                        help="Number of `net` snapshots concatenated into the conf head, taken at "
                             "ABSOLUTE iteration positions (k/K of conf_iters_max). K=1 -> one net; "
                             "K>1 -> feeds the refinement trajectory in feature space.")
    parser.add_argument('--conf_iters_max', type=int, default=12,
                        help="Cap for conf-head snapshot iterations (= train_iters). Ensures train "
                             "(12 iters) and test (24 iters) feed the head IDENTICAL states, since "
                             "RAFT's recurrence is deterministic and 13-24 are just continuation.")
    parser.add_argument('--conf_attenuate', action='store_true',
                        help="EGFS-faithful JOINT training: confidence weights the (final-iter) flow "
                             "loss [c*r_hat - alpha*log c], non-detached, so flow and confidence "
                             "co-adapt. Without it, the conf head is detached (a passenger).")
    parser.add_argument('--conf_no_tanh', action='store_true',
                        help="Use linear robustly-clamped r_hat instead of tanh (preserves "
                             "full-range ranking; avoids tail saturation).")
    parser.add_argument('--conf_clamp', type=float, default=3.0,
                        help="Max value of linear r_hat when --conf_no_tanh (robustness clamp).")
    parser.add_argument('--pose_e2e', action='store_true',
                        help="STAGE 5: end-to-end pose supervision. Adds an EPro-PnP Monte-Carlo pose "
                             "loss over DIFFERENTIABLE flow correspondences (x2d=source+flow, x3d=deproj, "
                             "w2d=softplus(conf)). Trains flow+conf jointly to be pose-relevant & "
                             "depth-UNbiased. Requires --conf_head. Plain flow loss is kept as anchor; "
                             "the attenuated/detached conf-loss branches are bypassed. With --freeze_main "
                             "= frozen-flow head only; without = full end-to-end.")
    parser.add_argument('--pose_e2e_weight', type=float, default=1.0,
                        help="Weight lambda of the MC pose loss added to the total loss.")
    parser.add_argument('--pose_e2e_npts', type=int, default=1024,
                        help="Correspondences subsampled per frame for the pose loss (EPro-PnP uses 512).")
    parser.add_argument('--pose_e2e_reg', type=float, default=1.0,
                        help="Weight of the pose-regression term (Huber trans + quat rot) inside the MC loss.")
    parser.add_argument('--pose_e2e_mc', type=int, default=512,
                        help="Monte-Carlo samples for the pose distribution (multiple of 4).")
    parser.add_argument('--pose_e2e_warmup', type=int, default=500,
                        help="Linearly ramp the pose-loss weight from 0 to --pose_e2e_weight over this "
                             "many steps, so the flow stabilizes under its own loss before the pose loss engages.")
    parser.add_argument('--pose_e2e_beta_pred', type=float, default=1.0,
                        help="Sweep A: relative weight on the MC loss log-partition term L_pred (the "
                             "parallax/geometry term) vs L_tgt (depth-biasing). >1 fights depth-bias. 1.0=orig.")
    parser.add_argument('--pose_e2e_alpha_tgt', type=float, default=1.0,
                        help="Weight on the MC loss target term L_tgt (depth-biasing reproj-at-GT cost). "
                             "0.0 = drop L_tgt entirely (L_pred-only). 1.0=orig.")
    parser.add_argument('--pose_e2e_reweight_temp', type=float, default=1.0,
                        help="Sweep B (decoupled only): temperature on the conf flow-reweighting "
                             "w=softplus(conf)^T. >1 sharpens conf's steer of the flow, <1 flattens toward "
                             "uniform. 1.0=orig decoupled behavior.")
    parser.add_argument('--pose_e2e_decouple', action='store_true',
                        help="STAGE 5 idea-2: DECOUPLED. Pose loss trains ONLY the conf head (flow detached "
                             "in the correspondence); the flow is trained by a conf-REWEIGHTED flow loss "
                             "(focus on the pose-supervised high-conf regions). Inherently stable (flow never "
                             "sees the raw pose gradient). Without this flag, --pose_e2e is full JOINT training.")
    parser.add_argument('--zero_flow', action='store_true',
                        help="ZERO-FLOW BRANCH (I2D-LocX): run a 2nd forward pass with the SAME shared "
                             "weights on the GT-pose (aligned) depth (depth_input ch2), supervised to "
                             "flow=0 over the aligned-depth projection mask (~15-20%%). Denser + more "
                             "direct cross-modal feature-matching signal; 0 extra params; discarded at "
                             "inference. Opt-in; composes with the main flow/edge/pose losses.")
    parser.add_argument('--zero_flow_weight', type=float, default=0.43,
                        help="Weight of the zero-flow loss term. Main flow loss coeff is 1.0, so 0.43 "
                             "reproduces I2D-LocX's lambda=0.7 main / 0.3 zero balance. Keep <1 so the "
                             "nonzero main branch stays dominant (paper's anti-collapse rationale).")
    parser.add_argument('--zero_flow_conf', action='store_true',
                        help="ZERO-FLOW variant 1b: conf-WEIGHT the zero-flow loss the same way the "
                             "decoupled recipe reweights the main flow loss -- per-pixel w = detached "
                             "softplus(conf) from the zero pass (same --pose_e2e_reweight_temp). Focuses "
                             "aligned-matching on pose-relevant high-conf pixels. Currently wired for the "
                             "full-network branch (use WITHOUT --zero_flow_encoder_only).")
    parser.add_argument('--zero_flow_encoder_only', action='store_true',
                        help="ZERO-FLOW variant 2a: the zero-flow gradient trains ONLY the feature encoders "
                             "(fnet_lidar=F_D/F_ED, fnet_event=F_EV that build the correlation), NOT the "
                             "update block / context encoder / edge branch / conf head. Done via a SEPARATE "
                             "backward with all non-fnet params requires_grad=False and in eval() (no BN-stat "
                             "pollution). Requires --zero_flow. The ONLY change vs full-network (1a) is where "
                             "the zero-flow gradient goes.")
    parser.add_argument('--corr_match', action='store_true',
                        help="OPTION 3 (correlation-level cross-modal matching, GRU-free): encoders-only "
                             "2nd forward on the GT-pose (aligned) depth (ch2) + events; build the full "
                             "correlation and supervise its identity diagonal to be the row-max via softmax "
                             "CE (LoFTR/GMFlowNet-style). Direct one-hop signal to fnet_lidar (F_D) + "
                             "fnet_event (F_EV), NO GRU in the path (cheaper than --zero_flow). 0 extra "
                             "params; discarded at inference. Opt-in; composes with main flow/edge/pose losses.")
    parser.add_argument('--corr_match_weight', type=float, default=0.1,
                        help="Weight of the correlation-matching loss (softmax CE, different scale than EPE). "
                             "Start small so the main flow loss (coeff 1.0) stays dominant.")
    parser.add_argument('--corr_match_norm', action='store_true',
                        help="L2-normalize features before the correlation (cosine + temperature) instead of "
                             "RAFT's raw dot with 1/sqrt(C) scaling. Default off = supervise the ACTUAL volume "
                             "the flow head consumes.")
    parser.add_argument('--corr_match_temp', type=float, default=0.1,
                        help="Softmax temperature, only used with --corr_match_norm (cosine logits).")
    parser.add_argument('--corr_match_flow', action='store_true',
                        help="VERSION B: supervise the REAL (perturbed-input) correlation to match at the "
                             "GT-flow-shifted target q=p+flow_gt(p) (vs Version-A identity on aligned depth). "
                             "This shapes the actual volume the GRU reads. Use with --corr_match_norm "
                             "--corr_match_temp 0.1 (peaky/learnable softmax) and optionally --corr_match_soft.")
    parser.add_argument('--corr_match_soft', type=float, default=0.0,
                        help="Soft Gaussian label std (in 1/8 cells) around the matching target; 0 = hard "
                             "one-hot CE. >0 absorbs sub-pixel/rounding + 1-vs-thousands imbalance (soft-CE).")
    parser.add_argument('--corr_match_conf', action='store_true',
                        help="CONF-WEIGHTED corr_match: weight each supervised row's matching CE by the "
                             "DETACHED softplus(conf) from the main forward (downsampled to 1/8, same "
                             "--pose_e2e_reweight_temp as the decoupled recipe) -> matching pressure focuses "
                             "on pose-relevant high-conf pixels. Requires --corr_match and --conf_head. "
                             "Default (off) = uniform. Analog of zero-flow 1b vs 1a.")
    parser.add_argument('--freeze_main', action='store_true',
                        help="Freeze all LEAR weights and train ONLY the conf head (fast Plan-1 "
                             "test on a warm-started converged model).")
    parser.add_argument('--train_sequence',
                        type=str,
                        default=None,
                        help="If set, train ONLY on this sequence (per-scene isolation). "
                             "When omitted, falls back to scenarios.yaml then to all non-test sequences.")
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--epochs', 
                        default=100, 
                        type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--starting_epoch', 
                        default=0, 
                        type=int, 
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', 
                        default=2, 
                        type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', 
                        '--learning_rate', 
                        default=4e-5, 
                        type=float,
                        metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--wdecay', 
                        type=float, 
                        default=.00005)
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=1e-8)
    parser.add_argument('--clip', 
                        type=float, 
                        default=1.0)
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.8, 
                        help='exponential weighting')
    parser.add_argument('--train_iters', 
                        type=int, 
                        default=12)
    parser.add_argument('--test_iters', 
                        type=int, 
                        default=24)
    parser.add_argument('--gpus', 
                        type=int, 
                        nargs='+', 
                        default=[0])
    parser.add_argument('--max_r', 
                        type=float, 
                        default=5.)
    parser.add_argument('--max_t', 
                        type=float, 
                        default=0.5)
    parser.add_argument('--max_depth',
                        type=float,
                        default=None,
                        help="depth normalization ceiling. Default None -> auto from scene_config "
                             "(indoor 10 / outdoor 100 / DSEC 50). Pass a value to override.")
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=3)
    parser.add_argument('--mixed_precision', 
                        action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--evaluate_interval',
                        default=1,
                        type=int,
                        metavar='N',
                        help='Evaluate every \'evaluate interval\' epochs ')
    parser.add_argument('--save_every_epochs',
                        default=0,
                        type=int,
                        metavar='N',
                        help='Also save a persistent checkpoint_ep{N}.pth every N epochs (0=off).')
    parser.add_argument('-e', 
                        '--evaluate', 
                        dest='evaluate', 
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--backbone',
                        type=str,
                        default='baseline')
    parser.add_argument('--use_feature_fusion', 
                        action='store_true')
    parser.add_argument('--bin', 
                        action='store_true')
    parser.add_argument('--save_log', 
                        action='store_true')
    parser.add_argument('--iteration_num',
                        default=1,
                        type=int)
    parser.add_argument('--seed',
                        default=1234,
                        type=int,
                        help='random seed for replicate runs (default 1234 = original)')
    args = parser.parse_args()

    ## scene_config: resolve crop + max_depth for this sequence (single source of truth).
    args._scene_ds = args.dataset
    DS = DatasetDSEC if args.dataset == 'dsec' else DatasetM3ED
    _g = scene_geometry(args.test_sequence, args._scene_ds)
    args._scene_crop = _g['crop']                       # (crop_h, crop_w, crop_x, crop_y)
    if args.max_depth is None:
        args.max_depth = _g['max_depth']
    print(f"[scene_config] seq={args.test_sequence} class={_g['klass']} "
          f"crop={args._scene_crop} calib_div={_g['calib_div']} max_depth={args.max_depth}", flush=True)

    ## set key parameters
    occlusion_kernel = 5
    occlusion_threshold = 3
    seed = args.seed
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])
    batch_size = args.batch_size

    ## initialize using fixed seed
    _init_fn(0, seed)

    if args.backbone == "baseline":
        model = torch.nn.DataParallel(Backbone_Event(args), device_ids=args.gpus)
    elif args.backbone == "edge":
        if args.use_feature_fusion:
            model = torch.nn.DataParallel(Backbone_Edge_FF(args), device_ids=args.gpus)
        else:
            model = torch.nn.DataParallel(Backbone_Edge(args), device_ids=args.gpus)
    else:
        raise "Specified backbone doesn't exist"
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        # strict=False so a warm-start from a pre-conf-head checkpoint keeps all LEAR weights
        # and only the new conf_head starts fresh (reported below). Additionally DROP any key whose
        # shape mismatches the current model (e.g. a flow-only phase saved a DEFAULT conf head that
        # differs from this run's conf_feats/kernel config) -> that head stays fresh instead of crashing.
        _sd = torch.load(args.load_checkpoints)
        _msd = model.state_dict()
        _mismatch = [k for k, v in _sd.items() if k in _msd and v.shape != _msd[k].shape]
        # if a differently-configured conf_head is dropped, also drop its stale conf_cfg buffer
        # (same shape, so not caught above) so it re-inits to THIS run's config -- otherwise the
        # saved conf_cfg is wrong and build_model() mis-infers the head at eval time.
        if any("conf_head" in k for k in _mismatch):
            _mismatch += [k for k in _sd if "conf_cfg" in k and k not in _mismatch]
        if _mismatch:
            print(f"[warm-start] dropping shape-mismatched/stale keys (kept fresh): {_mismatch}")
            _sd = {k: v for k, v in _sd.items() if k not in _mismatch}
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        if missing or unexpected:
            print(f"[warm-start] missing keys (fresh): {missing}\n[warm-start] unexpected keys: {unexpected}")
    if args.conf_head and args.freeze_main:
        nf = 0
        for name, p in model.named_parameters():
            if 'conf_head' not in name:
                p.requires_grad_(False); nf += 1
        print(f"[conf_head] main model FROZEN ({nf} tensors); training conf_head only")
    model.to(device)

    ## reinitialize using fixed seed
    _init_fn(0, seed)

    def init_fn(x):
        return _init_fn(x, seed)

    dataset_test = DS(args.data_path,
                      event_representation=args.ev_input,
                      max_r=args.max_r,
                      max_t=args.max_t,
                      split='test',
                      test_sequence=args.test_sequence)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time, epe, f1, pose_loss, inlier_rate = test(args, TestImgLoader, model, device, 
                                                                   occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold, 
                                                                   is_test=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"epe {epe:.5f} pose_loss {pose_loss:.5f} Mean {Time / len(TestImgLoader):.5f} per frame")
            print(f"inlier rate {inlier_rate/len(TestImgLoader):.5f}")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)} {outliers}")
        sys.exit()

    if args.dataset == 'dsec':
        # DatasetDSEC = single model on ALL train seqs; simpler signature (no train_sequence/edge_gt).
        dataset_train = DS(args.data_path,
                           event_representation=args.ev_input,
                           max_r=args.max_r,
                           max_t=args.max_t,
                           split='train',
                           test_sequence=args.test_sequence)
    else:
        dataset_train = DS(args.data_path,
                           event_representation=args.ev_input,
                           max_r=args.max_r,
                           max_t=args.max_t,
                           split='train',
                           test_sequence=args.test_sequence,
                           train_sequence=args.train_sequence,
                           edge_gt=args.edge_gt,
                           edge_gt_patch=args.edge_gt_patch,
                           edge_gt_tau=args.edge_gt_tau,
                           edge_gt_dilate=args.edge_gt_dilate,
                           dual_cam=args.dual_cam)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=args.num_workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)
    print("Train length: ", len(TrainImgLoader))
    print("Test length: ", len(TestImgLoader))

    optimizer, scheduler = fetch_optimizer(args, len(TrainImgLoader), model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, SUM_FREQ=100)

    datetime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    _tag = f'{args.ev_input}__{args.run_tag}' if args.run_tag else args.ev_input
    # --ckpt_dir override: fixed dir shared across resume windows (else a fresh datetime dir).
    ckpt_dir = args.ckpt_dir if getattr(args, 'ckpt_dir', None) else \
               f'./checkpoints/{args.test_sequence}/{args.backbone}/{_tag}/{datetime}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    starting_epoch = args.starting_epoch
    if starting_epoch > 0:
        for i in range(starting_epoch * len(TrainImgLoader)):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        logger.total_steps = starting_epoch * len(TrainImgLoader)

    min_val_err = 9999.
    max_epochs = args.epochs
    # --stop_epoch: bound THIS run to exit cleanly before a SLURM walltime kill (for resume chaining).
    # OneCycleLR still spans the full --epochs; we just stop the loop early and resubmit the next window.
    run_until = min(max_epochs, args.stop_epoch) if getattr(args, 'stop_epoch', None) else max_epochs
    # STAGE 5: instantiate the pose loss ONCE (its MonteCarloPoseLoss norm_factor persists across epochs).
    pose_e2e_loss = None
    if getattr(args, 'pose_e2e', False):
        assert args.conf_head, "--pose_e2e requires --conf_head (the weight head)."
        crop = args._scene_crop  # scene_config crop (per-sequence), matches train()/test()
        pose_e2e_loss = PoseE2ELoss(crop, max_depth=args.max_depth, n_sub=args.pose_e2e_npts,
                                    mc_samples=args.pose_e2e_mc, reg_weight=args.pose_e2e_reg,
                                    beta_pred=args.pose_e2e_beta_pred, alpha_tgt=args.pose_e2e_alpha_tgt).to(device)
    for epoch in range(starting_epoch, run_until):
        train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch, occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold, pose_e2e_loss=pose_e2e_loss)

        torch.cuda.empty_cache()

        if epoch % args.evaluate_interval == 0:
            epe, f1, pose_loss = test(args, TestImgLoader, model, device, occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold)
            print("Validation M3ED: %f, %f, %f" % (epe, f1, pose_loss))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1, 'm3ed-poseloss': pose_loss}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint.pth")

            if pose_loss < min_val_err:
                min_val_err = pose_loss
                torch.save(model.state_dict(), f'{ckpt_dir}/best_model.pth')

            torch.cuda.empty_cache()

        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint_ep{epoch + 1:03d}.pth")
            print(f"[save] {ckpt_dir}/checkpoint_ep{epoch + 1:03d}.pth")