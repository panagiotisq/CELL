import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.update import BasicUpdateBlock, EdgeUpdateBlock
from core.extractor import BasicEncoder, Encoder_Edge_Fusion, Decoder_Edge
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils import coords_grid, upflow, feature_visualizer, visualize_distinctiveness_map

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class Backbone_Event(nn.Module):
    def __init__(self, args):
        super(Backbone_Event, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder(input_dim=2, output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder(input_dim=1, output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(input_dim=1, output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, idx=0):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init


        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions








class EdgeDetector(torch.nn.Module):
    def __init__(self):
        super(EdgeDetector, self).__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, tenInput):
        tenInput = tenInput * 255.0

        tenVggOne = self.netVggOne(tenInput)    # Bx64xHxW
        tenVggTwo = self.netVggTwo(tenVggOne)   # Bx128xH/2xW/2
        tenVggThr = self.netVggThr(tenVggTwo)   # Bx256xH/4xW/4
        tenVggFou = self.netVggFou(tenVggThr)   # Bx512xH/8xW/8
        tenVggFiv = self.netVggFiv(tenVggFou)   # Bx512xH/16xW/16

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))

class Backbone_Edge(nn.Module):
    def __init__(self, args):
            super(Backbone_Edge, self).__init__()
            self.args = args

            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.level = args.corr_levels
            self.radius = args.corr_radius

            if 'dropout' not in self.args:
                self.args.dropout = 0

            if 'alternate_corr' not in self.args:
                self.args.alternate_corr = False

            self.edge_detector = EdgeDetector()

            # feature network, context network, and update block
            self.fnet_event = BasicEncoder(input_dim=2, output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.fnet_lidar = BasicEncoder(input_dim=2, output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(input_dim=2, output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, idx=0):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # detector edge mask
        edge_mask = self.edge_detector(image1)
        # # edge concat depth
        image1 = torch.cat((image1, edge_mask), dim=1)

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, edge_mask
            
        return flow_predictions, edge_mask
    








class ConfHead(nn.Module):
    """Per-pixel confidence head reading the flow-residual hidden state `net`. Returns a raw
    logit; softplus applied downstream. kernel=3 -> mirrors FlowHead (adds 5x5 local mixing);
    kernel=1 -> pure per-correspondence readout of the local net vector (no extra spatial mixing,
    since `net` already encodes a large receptive field). Output channels unchanged."""
    def __init__(self, input_dim=128, hidden_dim=256, kernel=3):
        super(ConfHead, self).__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel, padding=pad)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel, padding=pad)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Backbone_Edge_FF(nn.Module):
    def __init__(self, args):
        super(Backbone_Edge_FF, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_lidar = Encoder_Edge_Fusion(input_dim=1, output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_event = BasicEncoder(input_dim=2, output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(input_dim=1, output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.edge_detector = Decoder_Edge()

        self.edge_update_block = EdgeUpdateBlock(self.args, hidden_dim=256, input_dim=256+256)

        self.enhance_corr_layer = nn.Conv2d(836, 324, kernel_size=1)

        # DETACHED confidence head (Plan A): reads the flow-residual hidden state(s) `net`,
        # predicts per-pixel confidence. net is detached in forward so its gradients never
        # reach the main model. kernel from args (3=LEAR-style, 1=per-correspondence).
        # conf_nets K>1 concatenates K snapshots of `net` taken at RELATIVE iteration
        # positions (k/K of iters) -> captures the refinement TRAJECTORY and stays consistent
        # whether the model runs 12 (train) or 24 (test) iterations.
        self.conf_nets = max(1, int(getattr(args, 'conf_nets', 1)))
        # RAFT's recurrence is deterministic: state at iteration k is identical whether the model
        # runs 12 or 24 iters (24 just CONTINUES past 12). So conf-head snapshots must be taken at
        # ABSOLUTE iteration indices, capped at conf_iters_max (= the training iters), otherwise the
        # head sees different states at train (12) vs test (24).
        self.conf_iters_max = max(1, int(getattr(args, 'conf_iters_max', 12)))
        # which features feed the conf head: comma list of {net, cnet, motion}. `net` = K GRU-state
        # snapshots (late/flow-constrained); `cnet` = the upstream context (freer to encode
        # confidence under co-adaptation); `motion` = the encoded corr+flow (final iter).
        _feats = str(getattr(args, 'conf_feats', 'net')).split(',')
        self.conf_use_net = 'net' in _feats
        self.conf_use_cnet = 'cnet' in _feats
        self.conf_use_motion = 'motion' in _feats
        _dim = (hdim * self.conf_nets if self.conf_use_net else 0) \
             + ((hdim + cdim) if self.conf_use_cnet else 0) \
             + (128 if self.conf_use_motion else 0)
        self.conf_head = ConfHead(_dim, kernel=getattr(args, 'conf_kernel', 3))
        # config saved in the checkpoint so eval (build_model) can reconstruct the head exactly
        self.register_buffer('conf_cfg', torch.tensor(
            [self.conf_nets, int(self.conf_use_net), int(self.conf_use_cnet),
             int(self.conf_use_motion), int(getattr(args, 'conf_kernel', 3))]))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    def downsample_flow(self, flow, scale_factor):
        """
        Downsamples an optical flow field.
        
        Args:
            flow: Tensor of shape [B, 2, H, W] representing the optical flow.
            scale_factor: Factor by which to downsample (e.g., 0.5 for halving resolution).
            
        Returns:
            Downsampled flow tensor.
        """
        downsampled_flow = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        downsampled_flow[:, 0, :, :] *= scale_factor
        downsampled_flow[:, 1, :, :] *= scale_factor
        
        return downsampled_flow

    def corr_match_features(self, image1, image2):
        """OPTION 3 helper: run ONLY the two feature encoders (no context net, no GRU loop) and
        return the raw fmaps that CorrBlock would consume. Used to build the correlation-matching
        aux loss on the GT-pose (aligned) pair. image1=depth(1ch), image2=event(2ch). Applies the
        same 2*x-1 input scaling + autocast + .float() as forward()'s feature stage, so the features
        are identical to what the main correlation is built from. Cheap: encoders only."""
        image1 = 2 * image1.contiguous() - 1.0
        image2 = 2 * image2.contiguous() - 1.0
        with autocast(enabled=self.args.mixed_precision):
            fmap1, _ = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)
        return fmap1.float(), fmap2.float()

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, idx=0, output_edge=True, output_conf=False, conf_detach=True):
        """
            Estimate optical flow between pair of frames
            image1: lidar_input
            image2: event_frame
        """
        hdim = self.hidden_dim
        cdim = self.context_dim

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # optional stage timing (off unless self._bench is set by a benchmark script)
        _bench = getattr(self, '_bench', False)
        if _bench:
            self._bev = {k: torch.cuda.Event(enable_timing=True)
                         for k in ('enc0', 'enc1', 'flow1', 'conf1')}
            self._bev['enc0'].record()

        # print(torch.max(image1), torch.min(image1))
        # print(torch.max(image2), torch.min(image2))

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, edge_feature_list = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)


        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            cnet_feat = cnet.float()                       # full 256-d context, for the conf head
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        if _bench: self._bev['enc1'].record()   # end of feature encoding / start of flow loop

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        edge_predictions = []
        # ABSOLUTE snapshot iterations, capped at conf_iters_max, so train (12 iters) and test
        # (24 iters) feed the conf head the SAME states (iters 13-24 are extra continuation only).
        net_snaps = []
        _base = min(iters, self.conf_iters_max)
        snap_at = sorted({max(0, int(round((k + 1) / self.conf_nets * _base)) - 1)
                          for k in range(self.conf_nets)}) if output_conf else []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            corr_add_edge = torch.cat((corr, edge_feature_list[4]), dim=1)
            corr = self.enhance_corr_layer(corr_add_edge)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

            # detect edge
            edge_feature_4 = edge_feature_list[4]
            edge_feature_4_net, edge_feature_4_inp = torch.split(edge_feature_4, [256, 256], dim=1)
            warped_event_feature = self.warp(fmap2, coords1 - coords0)
            edge_feature_4 = self.edge_update_block(edge_feature_4_net, edge_feature_4_inp, warped_event_feature)
            edge_feature_list[4] = edge_feature_4

            if output_conf and itr in snap_at:
                net_snaps.append(net.detach() if conf_detach else net)

            if output_edge:
                edge_mask = self.edge_detector(edge_feature_list)
                edge_predictions.append(edge_mask)

        if _bench: self._bev['flow1'].record()   # end of flow loop (before conf head)

        # DETACHED confidence: read the final flow-residual hidden state, no grad to main model.
        # 1/8-res logit -> bilinear upsample to full res.
        if output_conf:
            parts = []
            if self.conf_use_net:
                while len(net_snaps) < self.conf_nets:             # pad if iters < conf_nets
                    net_snaps.append(net_snaps[-1] if net_snaps else (net.detach() if conf_detach else net))
                parts += net_snaps[:self.conf_nets]
            if self.conf_use_cnet:                                 # upstream context (freer to co-adapt)
                parts.append(cnet_feat.detach() if conf_detach else cnet_feat)
            if self.conf_use_motion:                               # encoded corr+flow, final iter
                mf = self.update_block._motion.float()
                parts.append(mf.detach() if conf_detach else mf)
            conf_logit = self.conf_head(torch.cat(parts, dim=1))
            conf_up = F.interpolate(conf_logit, scale_factor=8, mode='bilinear', align_corners=True)

        if _bench: self._bev['conf1'].record()   # end of confidence head

        if test_mode:
            if output_edge:
                rets = (flow_predictions, flow_up, edge_predictions)
            else:
                rets = (flow_predictions, flow_up)
        else:
            rets = (flow_predictions, edge_predictions)

        if output_conf:
            rets = rets + (conf_up,)
        return rets