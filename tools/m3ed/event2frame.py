import os

import sys
sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm

import h5py
import numpy as np
import torch
import cv2
import argparse
from utils import load_data, find_near_index
from tqdm import tqdm
from event_utils import events_to_voxel_timesync_torch

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/path/to/M3ED",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="falcon_indoor_flight_3",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--camera",
                    default="left",
                    help="which camera to use",
                    type=str)
    ap.add_argument("--method",
                    default="timesurface",
                    help="Event representation method",
                    type=str)
    ap.add_argument("--time_window",
                    default="100000",
                    help="length of time window",
                    type=int)
    ap.add_argument("--save_dir",
                    default="data/m3ed",
                    help="Path to save preprocessed data",
                    type=str)
    ap.add_argument("--half_resolution",
                      action='store_true')
    args = ap.parse_args()


    root = os.path.join(args.dataset, args.sequence)
    h5_file = f"{args.sequence}_data.h5"
    pose_file = f"{args.sequence}_pose_gt.h5"

    data_path = os.path.join(root, h5_file)
    pose_path = os.path.join(root, pose_file)

    data = h5py.File(data_path,'r')
    event_data_ref = load_data(data, sensor='prophesee', camera='left')
    event_data = load_data(data, sensor='prophesee', camera=args.camera)
    poses = h5py.File(pose_path,'r')
    ts_map_prophesee_left = poses['ts_map_prophesee_left']

    if not args.half_resolution:
        out_file = os.path.join(args.save_dir, args.sequence, f"event_frames_{args.method}_pre_{args.time_window}", args.camera)
    else:
        out_file = os.path.join(args.save_dir, args.sequence, f"event_frames_{args.method}_pre_{args.time_window}_half", args.camera)
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    
    if not args.half_resolution:
        k = 1
    else:
        k = 2

    rows, cols = event_data['resolution'][1]//k, event_data['resolution'][0]//k

    t_start = event_data['t'][0]

    for i in tqdm(range(len(ts_map_prophesee_left)-2)):
        idx_cur = int(ts_map_prophesee_left[i+1])
        t_ref = event_data_ref['t'][idx_cur]    # current timestamp

        idx_start, idx_cur, idx_end = find_near_index(event_data_ref['t'][idx_cur], event_data['t'], time_window=args.time_window * 2)

        event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

        if os.path.exists(f"{out_file}/event_frame_{i:05d}.npy"):
            continue
        
        if args.method == "timesurface":
            for id in range(idx_cur - idx_start):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
                if event_data['p'][idx] > 0:
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
        elif args.method == "ours_denoise_stc_trail":
            trail_filter_ths = args.time_window // 10  # Length of the time window for activity filtering (in us)
            trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)

            stc_filter_ths = args.time_window  // 10   # Length of the time window for filtering (in us)
            stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
            stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)

            events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

            new_dtype = np.dtype({
                            'names': ['x', 'y', 'p', 't'],                # Field names
                            'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
                            'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
                            'itemsize': 16                                # Total size of each entry
                            })
            events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
            events['x'] = event_data['x'][idx_start:idx_cur]
            events['y'] = event_data['y'][idx_start:idx_cur]
            events['p'] = event_data['p'][idx_start:idx_cur]
            events['t'] = event_data['t'][idx_start:idx_cur]

            stc_filter.process_events(events, events_buf)
            events_buf = np.asarray(events_buf)
            events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
            trail_filter.process_events(events_buf, events_buf_)
            events_buf = np.asarray(events_buf_)

            r = 6
            for idx in range(len(events_buf)):
                y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
                if events_buf['p'][idx] > 0:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    event_time_image[y, x, 0] = events_buf['t'][idx]
                else:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    event_time_image[y, x, 1] = events_buf['t'][idx]    
            if len(events_buf) > 0:        
                event_time_image[event_time_image > 0] -= events_buf['t'][0]
                event_time_image[event_time_image < 0] = 0           
        elif args.method == "bins":
            event_time_image = events_to_voxel_timesync_torch(torch.tensor(np.array(event_data['x'][idx_start:idx_cur]//k, dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['y'][idx_start:idx_cur]//k, dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['t'][idx_start:idx_cur], dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['p'][idx_start:idx_cur], dtype=np.float32)), 
                                                                B=10,
                                                                t0=event_data['t'][idx_start],
                                                                t1=event_data['t'][idx_cur],
                                                                np_ts=np.array(event_data['t'][idx_start:idx_cur], dtype=np.float32),
                                                                sensor_size=[rows, cols], temporal_bilinear=True)
        elif args.method == "SITS":
            r=6
            for id in range(idx_cur-idx_start):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
                if event_data['p'][idx] > 0:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 0]
                    patch = np.where(patch>=event_time_image[y,x,0], patch-1, patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 0] = patch
                    event_time_image[y, x, 0] = (2 * r + 1)**2
                else:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 1]
                    patch = np.where(patch>=event_time_image[y,x,1], patch-1, patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 1] = patch
                    event_time_image[y, x, 1] = (2 * r + 1)**2
            event_time_image[event_time_image < 0] = 0
        else:
            raise "Method doesn't exit."


        now = np.array(event_time_image)
        np.save(f"{out_file}/event_frame_{i:05d}", now)


