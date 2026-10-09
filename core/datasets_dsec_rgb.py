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
from torchvision import transforms
import torchvision.transforms.functional as F
from core.utils_point import quaternion_from_matrix, invert_pose, rotate_forward, rotate_back

def get_calib(sequence):
    if sequence in ['interlaken_00_c', 'interlaken_00_d', 'interlaken_00_e', 'interlaken_00_f', 'interlaken_00_g', 'thun_00_a']:
        return torch.tensor([1164.6238115833075, 1164.6238115833075, 713.5791168212891, 570.9349365234375])
    elif sequence in ['zurich_city_00_a', 'zurich_city_00_b', 'zurich_city_01_a', 'zurich_city_01_b', 'zurich_city_01_c', 
                      'zurich_city_01_d', 'zurich_city_01_e', 'zurich_city_01_f', 'zurich_city_02_a', 'zurich_city_02_b',
                      'zurich_city_02_c', 'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a']:
        return torch.tensor([1150.8249465165975, 1150.8249465165975, 724.4121398925781, 569.1058044433594])
    elif sequence in ['zurich_city_04_a',
                      'zurich_city_04_b', 'zurich_city_04_c', 'zurich_city_04_d', 'zurich_city_04_e', 'zurich_city_04_f',
                      'zurich_city_05_a', 'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a',
                      'zurich_city_09_a', 'zurich_city_09_b', 'zurich_city_09_c', 'zurich_city_09_d', 'zurich_city_09_e',
                      'zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_11_a', 'zurich_city_11_b', 'zurich_city_11_c']:
        return torch.tensor([1150.8943600390282, 1150.8943600390282, 723.4334411621094, 572.102180480957])
    else:
        raise "Sequence doesn't exist."

class DatasetDSECRGB(Dataset):
    def __init__(self, dataset_dir, max_t=0.5, max_r=5., split='test', device='cuda:0', test_sequence='thun_00_a'):
        super(DatasetDSECRGB, self).__init__()
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.split = split
        self.timestamps_list = {}

        self.all_files = []
        
        scene_list = [  
                        # training set
                        'interlaken_00_c',
                        'interlaken_00_d',
                        'interlaken_00_e',
                        'interlaken_00_f',
                        # 'interlaken_00_g',
                        'thun_00_a',
                        'zurich_city_00_a',
                        # 'zurich_city_00_b',
                        'zurich_city_01_a',
                        'zurich_city_01_b',
                        'zurich_city_01_c',
                        'zurich_city_01_d',
                        'zurich_city_01_e',
                        # 'zurich_city_01_f',
                        'zurich_city_02_a',
                        'zurich_city_02_b',
                        'zurich_city_02_c',
                        'zurich_city_02_d',
                        # 'zurich_city_02_e',
                        # 'zurich_city_03_a',
                        'zurich_city_04_a',
                        'zurich_city_04_b',
                        'zurich_city_04_c',
                        'zurich_city_04_d',
                        'zurich_city_04_e',
                        # 'zurich_city_04_f',
                        'zurich_city_05_a',
                        # 'zurich_city_05_b',
                        # 'zurich_city_06_a',
                        # 'zurich_city_07_a',
                        # 'zurich_city_08_a',
                        'zurich_city_09_a',
                        'zurich_city_09_b',
                        'zurich_city_09_c',
                        'zurich_city_09_d',
                        # 'zurich_city_09_e',
                        'zurich_city_10_a',
                        # 'zurich_city_10_b',
                        'zurich_city_11_a',
                        'zurich_city_11_b',
                        'zurich_city_11_c',
                     ]
        
        for dir in scene_list:
            self.timestamps_list[dir] = []

            timestamp_file = os.path.join(self.root_dir[:-10], "original", dir, f"{dir}_disparity_timestamps.txt")
            timestamps = np.loadtxt(timestamp_file, dtype='int64')
            timestamps = timestamps[1:]

            for idx in range(timestamps.size):
                if not os.path.exists(os.path.join(self.root_dir, dir, "local_maps_image", f"point_cloud_{idx*2:05d}"+'.h5')):
                    continue
                if not os.path.exists(os.path.join(self.root_dir, dir, "rgb", f"{idx*2:06d}"+'.png')):
                    continue
                if dir == test_sequence and split.startswith('test'):
                    self.all_files.append(os.path.join(dir, 'rgb', f"{idx*2:06d}"))
                elif (not dir == test_sequence) and split == 'train':
                    self.all_files.append(os.path.join(dir, "rgb", f"{idx*2:06d}"))
            
            self.timestamps_list[dir] = timestamps
        
        self.test_RT = []
        if split == 'test':
            test_RT_file = os.path.join(self.root_dir, f'test_RT_seq_{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
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
    
    def custom_transform(self, rgb, img_rotation=0., flip=False):
        resize = transforms.Resize((rgb.height // 2, rgb.width // 2))

        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = F.hflip(rgb)
            rgb = F.rotate(rgb, img_rotation)

        rgb = resize(rgb)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)

        return rgb
    
    def __len__(self):
        return len(self.all_files)    

    def __getitem__(self, idx):
        item = self.all_files[idx]
        run = str(item.split('/')[0])
        timestamp = str(item.split('/')[2])

        img_path = os.path.join(self.root_dir, run, "rgb", f'{timestamp}.png')
        pc_path = os.path.join(self.root_dir, run, "local_maps_image", "point_cloud_"+f"{int(timestamp):05d}"+'.h5')

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
        
        img = Image.open(img_path)

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[0, :] *= -1
        
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)

        img = self.custom_transform(img, img_rotation, h_mirror)

        if self.split == 'train':
            R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

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

        calib = get_calib(run)
        calib /= 2.

        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]


        sample = {'event_frame': img, 'point_cloud': pc_in, 
                  'calib': calib, 'tr_error': T, 'rot_error': R}

        return sample        