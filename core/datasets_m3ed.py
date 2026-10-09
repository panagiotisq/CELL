import os
import csv
import h5py
from math import radians
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import mathutils
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from core.utils_point import quaternion_from_matrix, invert_pose, rotate_forward, rotate_back
from core.scene_config import scene_geometry, assert_frame_resolution

def get_calib_m3ed(sequence, camera):
    if camera == "left":
        if sequence == "falcon_indoor_flight_1":
            return torch.tensor([1034.86278431, 1033.47800271, 629.70125104, 357.60071019])
        elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
            return torch.tensor([1034.39696079, 1034.73607278, 636.27410756, 364.73952748])
        elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
            return torch.tensor([1033.55774353, 1031.89258948, 632.18119829, 358.82456763])
        elif sequence in ['falcon_outdoor_night_high_beams']:
            return torch.tensor([1034.39696079, 1034.73607278,  636.27410756,  364.73952748])
        elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
            return torch.tensor([1034.39696079, 1034.73607278,  636.27410756,  364.73952748])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1033.22946467, 1032.37940162 , 637.14899526,  359.16366857]) 
        elif sequence in ['falcon_outdoor_day_fast_flight_1', 'falcon_outdoor_day_fast_flight_2']:
            return torch.tensor([1033.39370725, 1031.9143859,   636.02701332,  359.2474613 ])
        elif sequence in ['spot_indoor_building_loop', 'spot_outdoor_day_penno_short_loop']:
            return torch.tensor([1032.02301691, 1031.92283509,  635.79846694,  363.79830416])
        elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_2']:
            return torch.tensor([1031.52186613, 1031.70276624,  633.53886647,  364.34446137])
        elif sequence in['spot_outdoor_night_penno_short_loop']:
            return torch.tensor([1037.19386674, 1035.8444213,   629.4409195,   368.0727264])
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.96138833, 1032.8263147,   632.38290823, 368.57212974])
        elif sequence in ['spot_forest_easy_1', 'spot_forest_easy_2']:
            return torch.tensor([1031.96208381, 1031.83221391,  637.97533213,  367.507214  ])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.88399229, 1031.48192315 , 634.29808475 , 366.39105342])
        elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
            return torch.tensor([1031.84262859, 1030.10381777 , 635.71409589 , 365.59991372])
        elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
            return torch.tensor([1031.28303548, 1031.36293954,  635.73771622,  366.81925393])
        else:
            raise TypeError("Sequence Not Available")
    else:
        if sequence == "falcon_indoor_flight_1":
            return torch.tensor([1035.07712803, 1034.76733944,  632.04513322,  359.48878546])
        elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
            return torch.tensor([1034.61302587, 1034.83604567,  638.12992827,  366.88002829])
        elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
            return torch.tensor([1033.22781771, 1032.05548869, 631.84536312, 360.7175681])
        elif sequence in ['falcon_outdoor_night_high_beams']:
            return torch.tensor([1034.61302587, 1034.83604567,  638.12992827,  366.88002829])
        elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
            return torch.tensor([1034.61302587, 1034.83604567,  638.12992827,  366.88002829])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1032.7719854,  1031.93236836,  639.10742672,  362.60248038]) 
        elif sequence in ['falcon_outdoor_day_fast_flight_1', 'falcon_outdoor_day_fast_flight_2']:
            return torch.tensor([1032.16627257, 1030.86961564,  636.63418984,  362.53652701])
        elif sequence in ['spot_indoor_building_loop', 'spot_outdoor_day_penno_short_loop']:
            return torch.tensor([1031.49412912, 1031.36825419,  637.21356149,  365.77813682])
        elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_2']:
            return torch.tensor([1030.29161359, 1030.9024083, 634.79835424, 368.11576903])
        elif sequence in['spot_outdoor_night_penno_short_loop']:
            return torch.tensor([1036.12637943, 1035.52022498,  630.28001113,  370.22438737])
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.26867081, 1032.57448492,  632.85489859,  370.0490076 ])
        elif sequence in ['spot_forest_easy_1', 'spot_forest_easy_2']:
            return torch.tensor([1031.19326792, 1030.99426672,  639.38951597,  368.77569497])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.36879978 ,1031.06491961 , 634.87768084 , 367.62546105])
        elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
            return torch.tensor([1030.46186128, 1029.51180204,  635.69022466 , 364.32444857])
        elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
            return torch.tensor([1030.62646968, 1030.98305576,  636.68872098,  368.70448786])
        else:
            raise TypeError("Sequence Not Available")

def get_left_right_T(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 9.9999e-01, -1.2158e-04, -4.6864e-03, -1.2023e-01],
                             [ 1.2021e-04,  1.0000e+00, -2.9335e-04,  8.3630e-04],
                             [ 4.6865e-03,  2.9279e-04,  9.9999e-01,  1.0119e-03],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[ 9.9999e-01, -6.6613e-04, -3.5103e-03, -1.2018e-01],
                             [ 6.6661e-04,  1.0000e+00,  1.3561e-04,  9.1033e-04],
                             [ 3.5102e-03, -1.3795e-04,  9.9999e-01, -4.3059e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([[ 9.9999e-01, -3.8986e-04, -3.2093e-03, -1.2005e-01],
                             [ 3.8672e-04,  1.0000e+00, -9.8024e-04,  9.3651e-04],
                             [ 3.2097e-03,  9.7899e-04,  9.9999e-01,  3.4862e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_night_high_beams']:
        return torch.tensor([[ 9.99993617e-01, -6.66130268e-04, -3.51029598e-03, -1.20179395e-01],
                             [ 6.66610390e-04,  9.99999769e-01,  1.35607254e-04,  9.10326972e-04],
                             [ 3.51020483e-03, -1.37946389e-04,  9.99993830e-01, -4.30589681e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        return torch.tensor([[ 9.99993617e-01, -6.66130268e-04, -3.51029598e-03, -1.20179395e-01],
                             [ 6.66610390e-04,  9.99999769e-01,  1.35607254e-04,  9.10326972e-04],
                             [ 3.51020483e-03, -1.37946389e-04,  9.99993830e-01, -4.30589681e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['falcon_outdoor_day_fast_flight_1', 'falcon_outdoor_day_fast_flight_2']:
        return torch.tensor([[ 9.99992881e-01, -3.47893898e-04, -3.75720273e-03, -1.20145657e-01],
                             [ 3.39769023e-04,  9.99997603e-01, -2.16290172e-03,  9.08390044e-04],
                             [ 3.75794618e-03,  2.16160974e-03,  9.99990603e-01, -1.09685889e-03],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
        return torch.tensor([[ 9.99989586e-01, -4.60304775e-04, -4.54058647e-03 ,-1.20070481e-01],
                             [ 4.55317167e-04,  9.99999292e-01, -1.09942280e-03 , 9.50882179e-04],
                             [ 4.54108932e-03,  1.09734395e-03,  9.99989087e-01 ,-5.19321703e-04],
                             [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['spot_indoor_building_loop', 'spot_outdoor_day_penno_short_loop']:
        return torch.tensor([[ 9.99992307e-01, -5.40427065e-04, -3.88507544e-03, -1.20243363e-01],
                             [ 5.34746801e-04,  9.99998787e-01, -1.46296311e-03,  9.30707909e-04],
                             [ 3.88586135e-03,  1.46087433e-03,  9.99991383e-01, -3.13557943e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_2']:
        return torch.tensor([[ 9.9999e-01, -6.4771e-04, -3.6507e-03, -1.2011e-01],
                             [ 6.3675e-04,  1.0000e+00, -3.0024e-03,  8.7946e-04],
                             [ 3.6526e-03,  3.0000e-03,  9.9999e-01, -2.8266e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in['spot_outdoor_night_penno_short_loop']:
        return torch.tensor([[ 9.99994987e-01, -7.65895777e-04, -3.07222982e-03, -1.20177933e-01],
                             [ 7.61370080e-04,  9.99998624e-01, -1.47399785e-03,  9.47225178e-04],
                             [ 3.07335452e-03,  1.47165136e-03,  9.99994194e-01,  3.36550792e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['spot_forest_easy_1', 'spot_forest_easy_2']:
        return torch.tensor([[ 9.99990683e-01, -8.05185615e-04, -4.24103343e-03, -1.20256463e-01],
                             [ 7.97407623e-04,  9.99997998e-01, -1.83535678e-03,  9.27059444e-04],
                             [ 4.24250274e-03,  1.83195784e-03,  9.99989322e-01, -5.31172025e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
        return torch.tensor([[ 9.99992821e-01, -7.03218854e-04, -3.72345760e-03, -1.20137685e-01],
                             [ 6.95582202e-04,  9.99997653e-01, -2.05185517e-03,  1.03034216e-03],
                             [ 3.72489176e-03,  2.04925047e-03, 9.99990963e-01 , 4.66687603e-04],
                             [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00]])
    elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
        return torch.tensor([[ 9.99993926e-01 ,-7.53276578e-04, -3.40299857e-03, -1.20220978e-01],
                             [ 7.47230114e-04,  9.99998141e-01 ,-1.77772679e-03 , 9.05842201e-04],
                             [ 3.40433136e-03 , 1.77517317e-03,  9.99992630e-01, -1.13660539e-04],
                             [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
        return torch.tensor([[ 9.99996013e-01 ,-6.08160070e-04, -2.75745341e-03, -1.20188402e-01],
                             [ 6.09451349e-04,  9.99999705e-01,  4.67470872e-04 , 1.06054268e-03],
                             [ 2.75716830e-03, -4.69149542e-04,  9.99996089e-01,  1.07289756e-04],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])       
    elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
        return torch.tensor([[ 9.99992228e-01, -6.45914237e-04, -3.88937476e-03, -1.20192478e-01],
                             [ 6.37427554e-04,  9.99997414e-01, -2.18286148e-03 , 9.52772120e-04],
                             [ 3.89077465e-03,  2.18036532e-03,  9.99990054e-01,  5.21780332e-05],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]]) 
    else:
        raise TypeError("Sequence Not Available")

class DatasetM3ED(Dataset):
    def __init__(self, dataset_dir, event_representation, max_t=0.5, max_r=5., split='test', device='cuda:0', test_sequence='falcon_indoor_flight_3', train_sequence=None,
                 edge_gt=False, edge_gt_patch=3, edge_gt_tau=80, edge_gt_dilate=1, dual_cam=False,
                 test_RT_path=None):
        super(DatasetM3ED, self).__init__()
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.event_representation = event_representation
        self.split = split
        # SHARP EDGE GT: when on, __getitem__ computes an edge mask (extract_edges) on the
        # SAME processed/augmented event frame that push_fuse turns into event_mask, and
        # push_fuse restricts the depth2edge GT to depth ∩ (edge-events) instead of
        # depth ∩ (all events). Input frame is unchanged — only the GT's event-part is masked.
        self.edge_gt = edge_gt
        self.edge_gt_patch = edge_gt_patch
        self.edge_gt_tau = edge_gt_tau
        self.edge_gt_dilate = edge_gt_dilate
        self.GTs_R = {}
        self.GTs_T = {}

        self.all_files = []
        
        scene_list = [
                    'falcon_indoor_flight_1',
                    'falcon_indoor_flight_2',
                    'falcon_indoor_flight_3',
                    # 'falcon_outdoor_day_penno_parking_1',
                    # 'falcon_outdoor_day_penno_parking_2',
                    # 'falcon_outdoor_night_high_beams',
                    # 'falcon_outdoor_night_penno_parking_1',
                    # 'falcon_outdoor_night_penno_parking_2',
                    # "falcon_outdoor_day_fast_flight_1",
                    # "falcon_outdoor_day_fast_flight_2",
                    # 'falcon_forest_into_forest_1',
                    # 'falcon_forest_into_forest_2',
                    # 'falcon_forest_into_forest_4',
                    # 'spot_indoor_building_loop',
                    # 'spot_outdoor_day_penno_short_loop',
                    # 'spot_outdoor_night_penno_short_loop',
                    # 'spot_forest_easy_1',
                    # 'spot_forest_easy_2',
                    # 'spot_forest_road_1',
                    # 'spot_forest_road_3',
                    # 'spot_outdoor_day_srt_under_bridge_1',
                    # 'spot_outdoor_day_srt_under_bridge_2',
                    # 'car_urban_day_penno_big_loop',
                    # 'car_urban_day_penno_small_loop',
                    # 'car_urban_night_penno_big_loop',
                    # 'car_urban_night_penno_small_loop',
                    # 'car_forest_into_ponds_long',
                    # 'car_forest_into_ponds_short',
                     ]

        # Merge in any sequences registered in scenarios.yaml so new
        # scenes work without editing this list. (Falcon entries above stay as-is.)
        try:
            from core.scene_registry import registered_sequences
            for _s in registered_sequences():
                if _s not in scene_list:
                    scene_list.append(_s)
        except Exception:
            pass

        # Ensure the requested test sequence is always loadable, even if it isn't
        # registered in scenarios.yaml or listed above (needed for newly-added scenes).
        if split.startswith('test') and test_sequence is not None and test_sequence not in scene_list:
            scene_list.append(test_sequence)
        # Same for TRAIN: a newly-added train_sequence (not in scenarios.yaml / list above) must be
        # in scene_list or the train set is empty (num_samples=0). Mirrors the test tweak.
        if split == 'train' and train_sequence is not None:
            for _s in str(train_sequence).split(','):
                _s = _s.strip()
                if _s and _s not in scene_list:
                    scene_list.append(_s)

        # Per-scene isolation: train ONLY on the designated train sequence(s).
        # train_sequence may be a single name OR a comma-separated list — the paper
        # config for a multi-sequence scene (e.g. 'falcon_indoor_flight_1,falcon_indoor_flight_2'
        # trains on both, tests on flight_3).
        # Priority: explicit train_sequence arg > scenarios.yaml (by test_sequence) >
        # original "all non-test" behaviour (only when neither is available).
        train_set = None
        if train_sequence is not None:
            train_set = {s.strip() for s in str(train_sequence).split(',') if s.strip()}
        else:
            try:
                from core.scene_registry import scenario_for
                _, _sc = scenario_for(test_sequence)
                if _sc and _sc.get('train_seq'):
                    _ts = _sc.get('train_seq')
                    train_set = set(_ts) if isinstance(_ts, (list, tuple)) else {_ts}
            except Exception:
                pass

        for dir in scene_list:
            self.GTs_R[dir] = []
            self.GTs_T[dir] = []

            pose_path = os.path.join(self.root_dir, dir, dir+"_pose_gt.h5")
            poses = h5py.File(pose_path,'r')
            Ln_T_L0 = poses['Ln_T_L0'] 

            if dir in ['falcon_indoor_flight_1', 'falcon_indoor_flight_2', 'falcon_indoor_flight_3']:
                left = 70
                right = 70
            elif dir in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
                left = 100
                right = 100
            elif dir in ['falcon_outdoor_night_penno_parking_1']:
                left = 220
                right = 160
            elif dir in ['falcon_outdoor_night_penno_parking_2']:
                left = 112
                right = 127
            elif dir in ['falcon_outdoor_night_high_beams']:
                left = 280
                right = 100
            elif dir in ['falcon_forest_into_forest_1']:
                left = 440
                right = 480
            elif dir in ['falcon_forest_into_forest_2']:
                left = 240
                right = 230
            elif dir in ['falcon_forest_into_forest_4']:
                left =  100
                right = 380
            elif dir in ['spot_indoor_building_loop']:
                left =  40
                right = 20
            elif dir in ['spot_outdoor_day_penno_short_loop']:
                left =  31
                right = 100
            elif dir in ['spot_outdoor_night_penno_short_loop']:
                left =  60
                right = 240
            elif dir in ['spot_forest_easy_1']:
                left =  30
                right = 70
            elif dir in ['spot_forest_easy_2']:
                left = 36
                right = 50
            elif dir in ['spot_forest_road_1']:
                left =  35
                right = 360
            elif dir in ['spot_forest_road_3']:
                left = 115
                right = 370
            elif dir in ['car_urban_day_penno_big_loop']:
                left = 90
                right = 48
            elif dir in ['car_urban_day_penno_small_loop']:
                left = 70
                right = 20
            elif dir in ['car_urban_night_penno_big_loop']:
                left = 90
                right = 88
            elif dir in ['car_urban_night_penno_small_loop']:
                left = 60
                right = 40
            elif dir in ['car_forest_into_ponds_long']:
                left = 200
                right = 30
            elif dir in ['car_forest_into_ponds_short']:
                left = 40
                right = 180
            else:
                # Unknown scene: look up margins from scenarios.yaml
                from core.scene_registry import margins_for
                _m = margins_for(dir)
                if _m is None:
                    raise ValueError(
                        f"Sequence '{dir}' has no margins: add it to "
                        f"scenarios.yaml or the hardcoded list above."
                    )
                left, right = _m

            for idx in range(Ln_T_L0.shape[0]):
                if idx + right < Ln_T_L0.shape[0] and idx > left: 
                    if not os.path.exists(os.path.join(self.root_dir, dir, "local_maps", f"point_cloud_{idx:05d}"+'.h5')):
                        continue
                    if not os.path.exists(os.path.join(self.root_dir, dir, f"event_frames_{self.event_representation}", 'left', f"event_frame_{idx:05d}"+'.npy')):
                        continue
                    if dir in ["spot_indoor_building_loop", "spot_outdoor_day_penno_short_loop", "spot_outdoor_night_penno_short_loop", "falcon_outdoor_night_high_beams", "falcon_outdoor_night_penno_parking_1"]:
                        if split.startswith('test'):
                            if (idx - left) / (Ln_T_L0.shape[0] - left - right) > 0.7:
                                self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'left', f"{idx:05d}"))
                            else:
                                continue
                        elif split == 'train':
                            if (idx - left) / (Ln_T_L0.shape[0] - left - right) <= 0.7:
                                self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'left', f"{idx:05d}"))
                                # --dual_cam: also train on the right-camera frame (~2x data for these
                                # data-starved single-seq scenes). Right frames were generated for exactly
                                # these scenes; __getitem__ handles the right calib + stereo extrinsic. Test
                                # stays left-only (unchanged), so eval comparability is preserved.
                                if dual_cam and os.path.exists(os.path.join(self.root_dir, dir, f"event_frames_{self.event_representation}", 'right', f"event_frame_{idx:05d}.npy")):
                                    self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'right', f"{idx:05d}"))
                            else:
                                continue
                    else:
                        if dir == test_sequence and split.startswith('test'):
                            self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'left', f"{idx:05d}"))
                        elif split == 'train' and (dir in train_set if train_set else not dir == test_sequence):
                            self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'left', f"{idx:05d}"))
                            # right-camera frames: the hardcoded multi-seq subset always; ANY train seq under
                            # --dual_cam (e.g. car_urban_night trains on big_loop, which has right frames). Test stays left.
                            if dual_cam or dir in ['falcon_indoor_flight_1', 'falcon_indoor_flight_2', 'falcon_outdoor_day_penno_parking_1', 'spot_forest_road_1', 'spot_forest_easy_2']:
                                # only add right-camera frames if they were actually generated
                                if os.path.exists(os.path.join(self.root_dir, dir, f"event_frames_{self.event_representation}", 'right', f"event_frame_{idx:05d}"+'.npy')):
                                    self.all_files.append(os.path.join(dir, f"event_frames_{self.event_representation}", 'right', f"{idx:05d}"))
                    
                    R = quaternion_from_matrix(torch.tensor(Ln_T_L0[idx]))
                    T = Ln_T_L0[idx][:3, 3]
                    GT_R = np.asarray(R)
                    GT_T = np.asarray(T)
                    self.GTs_R[dir].append(GT_R)
                    self.GTs_T[dir].append(GT_T)
                else:
                    continue
        
        self.test_RT = []
        if split == 'test':
            # test_RT_path override (perturbation study): read this exact file instead of the
            # auto-constructed canonical name. Default None => identical original behavior.
            test_RT_file = test_RT_path if test_RT_path else os.path.join(self.root_dir, f'test_RT_seq_{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-min(max_t, 0.1), min(max_t, 0.1))
                    test_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])
                    self.test_RT.append([i, transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"
    
    
    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def __len__(self):
        return len(self.all_files)    

    def __getitem__(self, idx):
        item = self.all_files[idx]
        run = str(item.split('/')[0])
        camera = str(item.split('/')[2])
        timestamp = str(item.split('/')[3])

        event_frame_path = os.path.join(self.root_dir, run, f"event_frames_{self.event_representation}", camera, 'event_frame_'+timestamp+'.npy')
        pc_path = os.path.join(self.root_dir, run, "local_maps", "point_cloud_"+f"{int(timestamp)+1:05d}"+'.h5')

        try:
            with h5py.File(pc_path, 'r') as hf:
                pc = hf['PC'][:]
        except Exception as e:
            print(f'File Broken: {pc_path}')
            raise e
        
        pc_in = torch.from_numpy(pc.astype(np.float32))
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        if camera == 'right':
            T_to_prophesee_left = get_left_right_T(run)
            pc_in = torch.matmul(T_to_prophesee_left, pc_in)

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[0, :] *= -1
        
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)

        if self.split == 'train':
            R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        event_frame = np.load(event_frame_path)
        # SAFETY NET: verify this seq's actual frame resolution matches its scene_config class,
        # so a mis-generated/mis-classified sequence errors loudly instead of running wrong geometry.
        assert_frame_resolution(run, "m3ed", event_frame.shape[0], event_frame.shape[1])
        event_time_frame = torch.tensor(event_frame).float()
        event_time_frame[event_time_frame<0] = 0
        event_time_frame /= torch.max(event_time_frame)

        if event_time_frame.shape[2] <= 4:
            event_time_frame = F.to_pil_image(event_time_frame.permute(2, 0, 1))
            if h_mirror:
                event_time_frame = event_time_frame.transpose(Image.FLIP_LEFT_RIGHT)
            event_time_frame = event_time_frame.rotate(img_rotation)
            event_time_frame = F.to_tensor(event_time_frame)
            event_frame = event_time_frame
        else:
            event_frames = []
            for i in range(0, event_time_frame.shape[2], 4):
                event_time_frame_sub = F.to_pil_image(event_time_frame[:, :, i:i+4].permute(2, 0, 1))
                if h_mirror:
                    event_time_frame_sub = event_time_frame_sub.transpose(Image.FLIP_LEFT_RIGHT)
                event_time_frame_sub = event_time_frame_sub.rotate(img_rotation)
                event_time_frame_sub = F.to_tensor(event_time_frame_sub)
                event_frames.append(event_time_frame_sub)
            event_frame = torch.cat(event_frames, dim=0)

        if self.split != 'test':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-min(self.max_t, 0.1), min(self.max_t, 0.1))
        else:
            initial_RT = self.test_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        calib = get_calib_m3ed(run, camera)
        # scene_config: half-res seqs need calib/2 (native calib is for the 1280x720 sensor);
        # full-res night/fast seqs use native calib (calib_div=1). Single source of truth.
        calib /= scene_geometry(run, "m3ed")["calib_div"]

        if h_mirror:
            calib[2] = event_frame.shape[2] - calib[2]


        sample = {'event_frame': event_frame, 'point_cloud': pc_in,
                  'calib': calib, 'tr_error': T, 'rot_error': R}

        if self.edge_gt:
            # Edge mask on the SAME processed+augmented event frame that becomes push_fuse's
            # event_mask (full res, before the perturbation/rotation) -> frame-correct.
            from tools.pose_refine.edge_extract import extract_edges
            a = event_frame.numpy()
            twoch = np.stack([np.abs(a[0]), np.abs(a[1])], -1).astype(np.float32)  # [H,W,2]
            em = extract_edges(twoch, patch=self.edge_gt_patch, tau_pct=self.edge_gt_tau,
                               min_area=30, dilate=self.edge_gt_dilate)             # [H,W] {0,1}
            sample['edge_mask'] = torch.from_numpy(em).float()

        return sample