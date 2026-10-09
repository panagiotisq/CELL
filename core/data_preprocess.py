import torch
import visibility
import mathutils
import numpy as np
import cv2

from core.utils_point import rotate_back, rotate_forward, to_rotation_matrix
from core.camera_model import CameraModel
from core.depth_completion import sparse_to_dense

class Data_preprocess:
    def __init__(self, calibs, occlusion_threshold, occlusion_kernel, partial_fill=None):
        self.real_shape = None
        self.calibs = calibs
        self.occlusion_threshold = occlusion_threshold
        self.occlusion_kernel = occlusion_kernel
        # opt-in: fill_level for the ENCODER-INPUT dense channel (ch1): None => 'full' (unchanged);
        # 'partial' / 'partial_light' / 'partial_dc' select the experimental completions.
        # Default None => sparse (ch0) and full dense (ch1) behave exactly as before.
        self.partial_fill = partial_fill

    def delta_1(self, uv_RT, uv, VI_indexes_RT, VI_indexes):
        indexes = VI_indexes_RT & VI_indexes

        indexes_1 = indexes[VI_indexes_RT]
        indexes_2 = indexes[VI_indexes]

        delta_P = uv[indexes_2, :] - uv_RT[indexes_1, :]

        return delta_P, indexes

    def gen_depth_img(self, uv_RT_af_index, depth_RT_af_index, indexes_uvRT, cam_params):
        device = uv_RT_af_index.device

        depth_img_RT = torch.zeros(self.real_shape[:2], device=device, dtype=torch.float)
        depth_img_RT += 1000.

        idx_img = (-1) * torch.ones(self.real_shape[:2], device=device, dtype=torch.float)
        indexes_uvRT = indexes_uvRT.float()

        depth_img_RT, idx_img = visibility.depth_image(uv_RT_af_index, depth_RT_af_index, indexes_uvRT,
                                                       depth_img_RT, idx_img, uv_RT_af_index.shape[0],
                                                       self.real_shape[1], self.real_shape[0])
        depth_img_RT[depth_img_RT == 1000.] = 0.

        deoccl_index_img = (-1) * torch.ones(self.real_shape[:2], device=device, dtype=torch.float)

        depth_img_no_occlusion_RT = torch.zeros_like(depth_img_RT, device=device)
        depth_img_no_occlusion_RT, deoccl_index_img = visibility.visibility2(depth_img_RT, cam_params,
                                                                             idx_img,
                                                                             depth_img_no_occlusion_RT,
                                                                             deoccl_index_img,
                                                                             depth_img_RT.shape[1],
                                                                             depth_img_RT.shape[0],
                                                                             self.occlusion_threshold,
                                                                             int(self.occlusion_kernel))

        return depth_img_no_occlusion_RT, deoccl_index_img.int()

    def fresh_indexes(self, indexes_uvRT_deoccl, indexes_uvRT):
        indexes_uvRT_deoccl_list_indexes = torch.where(indexes_uvRT_deoccl > 0)

        indexes_uvRT_deoccl_list = indexes_uvRT_deoccl[indexes_uvRT_deoccl_list_indexes[0][:], indexes_uvRT_deoccl_list_indexes[1][:]]

        indexes_temp = torch.zeros(indexes_uvRT.shape[0], device=indexes_uvRT_deoccl_list.device, dtype=torch.int32)
        
        # indexes_temp[indexes_uvRT_deoccl_list.cpu().numpy() - 1] = indexes_uvRT_deoccl_list
        indexes = indexes_uvRT_deoccl_list.cpu().numpy() - 1
        indexes[indexes==indexes_temp.shape[0]] -= 1
        indexes_temp[indexes] = indexes_uvRT_deoccl_list


        return indexes_temp

    def delta_2(self, delta_P, uv_RT_af_index, mask):
        device = delta_P.device

        delta_P_com = delta_P[mask, :]

        delta_P_0 = delta_P_com[:, 0]
        delta_P_1 = delta_P_com[:, 1]

        ## keep common points after deocclusion
        uv_RT_af_index_com = uv_RT_af_index[mask, :]

        ## generate displacement map
        project_delta_P_1 = torch.zeros(self.real_shape[:2], device=device, dtype=torch.int32)
        project_delta_P_2 = torch.zeros(self.real_shape[:2], device=device, dtype=torch.int32)
        project_delta_P_1[uv_RT_af_index_com[:, 1].cpu().numpy(), uv_RT_af_index_com[:, 0].cpu().numpy()] = delta_P_0
        project_delta_P_2[uv_RT_af_index_com[:, 1].cpu().numpy(), uv_RT_af_index_com[:, 0].cpu().numpy()] = delta_P_1

        project_delta_P_shape = list(self.real_shape[:2])
        project_delta_P_shape.insert(0, 2)
        project_delta_P = torch.zeros(project_delta_P_shape, device=device, dtype=torch.float)

        project_delta_P[0, :, :] = project_delta_P_1
        project_delta_P[1, :, :] = project_delta_P_2

        return project_delta_P

    def DownsampleCrop_M3ED_delta(self, img, depth, displacement, split, h=600, w=960):
        if split == 'train':
            x = np.random.randint(0, img.shape[1] - h)
            y = np.random.randint(0, img.shape[2] - w)
        else:
            x = (img.shape[1] - h) // 2
            y = (img.shape[2] - w) // 2
        img = img[:, x:x + h, y:y + w]
        depth = depth[:, x:x + h, y:y + w]
        displacement = displacement[:, x:x + h, y:y + w]
        return img, depth, displacement
    
    def DownsampleCrop_M3ED_delta_mask(self, img, depth, displacement, mask, split, h=600, w=960):
        if split == 'train':
            x = np.random.randint(0, img.shape[1] - h)
            y = np.random.randint(0, img.shape[2] - w)
        else:
            x = (img.shape[1] - h) // 2
            y = (img.shape[2] - w) // 2
        img = img[:, x:x + h, y:y + w]
        depth = depth[:, x:x + h, y:y + w]
        displacement = displacement[:, x:x + h, y:y + w]
        mask = mask[:, x:x + h, y:y + w]
        return img, depth, displacement, mask
    
    def DownsampleCrop_input(self, img, depth, split, h=600, w=960):
        if split == 'train':
            x = np.random.randint(0, img.shape[1] - h)
            y = np.random.randint(0, img.shape[2] - w)
        else:
            x = (img.shape[1] - h) // 2
            y = (img.shape[2] - w) // 2
        img = img[:, x:x + h, y:y + w]
        depth = depth[:, x:x + h, y:y + w]
        return img, depth, x, y
    
    def DownsampleCrop_flow(self, displacement, x, y, downsample=None, h=600, w=960):
        if downsample is not None:
            displacement = displacement[:, x:x + h//downsample, y:y + w//downsample]
        else:
            displacement = displacement[:, x:x + h, y:y + w]
        return displacement
    

    def push(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train'):
        depth_input = []
        rgb_input = []
        flow_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            pc_rotated = rotate_back(pc, RT)    # Nx4

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, indexes_uvRT_deoccl_training = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)

            depth_img_no_occlusion_RT_training /= MAX_DEPTH

            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            rgb, depth_img_no_occlusion_RT_training, project_delta_P \
                = self.DownsampleCrop_M3ED_delta(rgb, depth_img_no_occlusion_RT_training, project_delta_P, split, h=h, w=w)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)

        return rgb_input, depth_input, flow_gt

    def push_dense_flow(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train'):
        depth_input = []
        rgb_input = []
        flow_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            pc_rotated = rotate_back(pc, RT)
            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()
            
            ## complete point cloud
            depth_img_no_occlusion_RT_training, _ = self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            depth_img_no_occlusion_RT_training_dense = sparse_to_dense(depth_img_no_occlusion_RT_training.cpu().detach().numpy(),
                fill_level=(self.partial_fill or 'full'))   # respect partial_light => dense flow on the plight support (one-variable exp)
            depth_img_no_occlusion_RT_training_dense = torch.tensor(depth_img_no_occlusion_RT_training_dense, device=device)
            pc_rotated = cam_model.depth2pc(depth_img_no_occlusion_RT_training_dense)
            pc_rotated = torch.tensor(pc_rotated, device=device)
            pc = rotate_forward(pc_rotated, RT)
            # uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            # uv = uv.t().int().contiguous()
            # depth_img, _ = self.gen_depth_img(uv, depth, VI_indexes[VI_indexes], cam_params)
            # depth_img = sparse_to_dense(depth_img.cpu().detach().numpy())
            # depth_img = torch.tensor(depth_img, device=device)
            # pc = cam_model.depth2pc(depth_img)
            # pc = torch.tensor(pc, device=device)
            # pc_rotated = rotate_back(pc, RT)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            # depth_img_no_occlusion_RT_training, _ = self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            depth_img_no_occlusion_RT_training /= MAX_DEPTH
            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            rgb, depth_img_no_occlusion_RT_training, project_delta_P \
                = self.DownsampleCrop_M3ED_delta(rgb, depth_img_no_occlusion_RT_training, project_delta_P, split, h=h, w=w)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)

        return rgb_input, depth_input, flow_gt

    def push_use_mask(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train'):
        depth_input = []
        rgb_input = []
        flow_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            pc_rotated = rotate_back(pc, RT)    # Nx4

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, indexes_uvRT_deoccl_training = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            
            
            # make depth image for generating depth mask
            depth_img_no_occlusion_GT, indexes_uvRT_deoccl_training = \
                self.gen_depth_img(uv, depth, VI_indexes[VI_indexes], cam_params)
            depth_img_no_occlusion_GT = sparse_to_dense(depth_img_no_occlusion_GT.cpu().detach().numpy())
            depth_img_no_occlusion_GT = torch.tensor(depth_img_no_occlusion_GT, device=device)
            event_mask = (rgb[0, :, :] > 0) + (rgb[1, :, :] > 0)
            depth_img_no_occlusion_GT_masked = depth_img_no_occlusion_GT * torch.tensor(event_mask, device=device)
            pc_masked = cam_model.depth2pc(depth_img_no_occlusion_GT_masked)
            pc_masked_rotated = rotate_back(torch.tensor(pc_masked, device=device), RT)
            uv_masked_RT, depth_masked_RT, _, _, VI_masked_indexes_RT = cam_model.project_withindex_pytorch(pc_masked_rotated, self.real_shape)
            uv_masked_RT = uv_masked_RT.t().int().contiguous()
            depth_img_mask_no_occlusion_RT_training, _ = self.gen_depth_img(uv_masked_RT, depth_masked_RT, VI_masked_indexes_RT[VI_masked_indexes_RT], cam_params)
            mask = depth_img_mask_no_occlusion_RT_training > 0

            # depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training * mask

            depth_img_no_occlusion_RT_training /= MAX_DEPTH
            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            rgb, depth_img_no_occlusion_RT_training, project_delta_P \
                = self.DownsampleCrop_M3ED_delta(rgb, depth_img_no_occlusion_RT_training, project_delta_P, split, h=h, w=w)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)

        return rgb_input, depth_input, flow_gt

    def push_fuse(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train', edge_masks=None, dense_flow=False):
        """
            output:
                    rgb_input:      Bx2xHxW
                    depth_input:    Bx3xHxW (depth_input, depth_input_dense, depth_gt)
                    flow_gt:        Bx2xHxW
                    depth_mask_gt:  Bx2xHxW (edge_mask, event_mask)

            dense_flow: if True, the flow target is a COMPOSITE — the exact sparse GT flow on real
            projected points, PLUS a geometrically-consistent flow (from the completed depth back-
            projected to 3D) on the newly-filled completion pixels. Everything else (both depth
            channels, the edge/depth-mask GT) is unchanged, so it is a strictly one-variable change
            (flow-supervision density) vs the plight input.
        """
        depth_input = []
        rgb_input = []
        flow_gt = []
        depth_mask_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            pc_rotated = rotate_back(pc, RT)
            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, indexes_uvRT_deoccl_training = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            depth_img_no_occlusion_RT_training_dense = sparse_to_dense(depth_img_no_occlusion_RT_training.cpu().detach().numpy(),
                fill_level=(self.partial_fill or 'full'))
            depth_img_no_occlusion_RT_training_dense = torch.tensor(depth_img_no_occlusion_RT_training_dense, device=device)
            dense_meters = depth_img_no_occlusion_RT_training_dense.clone() if dense_flow else None  # completed depth (m) for dense-flow target
            depth_img_no_occlusion_RT_training /= MAX_DEPTH
            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)
            depth_img_no_occlusion_RT_training_dense /= MAX_DEPTH
            depth_img_no_occlusion_RT_training_dense = depth_img_no_occlusion_RT_training_dense.unsqueeze(0)
            depth_img_no_occlusion_RT_training = torch.cat((depth_img_no_occlusion_RT_training, depth_img_no_occlusion_RT_training_dense), dim=0)

            # make depth image for generating depth mask
            depth_img_no_occlusion, _ = \
                self.gen_depth_img(uv, depth, VI_indexes[VI_indexes], cam_params)
            depth_img_no_occlusion_GT = sparse_to_dense(depth_img_no_occlusion.cpu().detach().numpy())
            depth_img_no_occlusion_GT = torch.tensor(depth_img_no_occlusion_GT, device=device)
            event_mask = (rgb[0, :, :] > 0) + (rgb[1, :, :] > 0)
            if edge_masks is not None:
                # SHARP EDGE GT: restrict events to their edges BEFORE the depth intersection
                # and rotation, in this (unperturbed, full-res) frame -> depth ∩ edge-events.
                event_mask = event_mask * edge_masks[idx].to(device)
            depth_img_no_occlusion_GT_masked = depth_img_no_occlusion_GT * event_mask
            pc_masked = cam_model.depth2pc(depth_img_no_occlusion_GT_masked)
            pc_masked_rotated = rotate_back(torch.tensor(pc_masked, device=device), RT)
            uv_masked_RT, depth_masked_RT, _, _, VI_masked_indexes_RT = cam_model.project_withindex_pytorch(pc_masked_rotated, self.real_shape)
            uv_masked_RT = uv_masked_RT.t().int().contiguous()
            depth_img_no_occlusion_masked_RT, _ = self.gen_depth_img(uv_masked_RT, depth_masked_RT, VI_masked_indexes_RT[VI_masked_indexes_RT], cam_params)
            depth_img_no_occlusion_masked_RT /= MAX_DEPTH
            depth_img_no_occlusion_masked_RT = depth_img_no_occlusion_masked_RT.unsqueeze(0)
            depth_mask = depth_img_no_occlusion_masked_RT>0
            depth_mask = torch.cat((depth_mask, event_mask.unsqueeze(0)), dim=0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            if dense_flow:
                # Dense flow over the COMPLETED support: back-project the completed depth to 3D
                # (perturbed frame), rotate to the GT frame, and recompute the displacement with the
                # SAME deocclusion pipeline (delta_1/gen_depth_img/fresh_indexes/delta_2). Then keep
                # the exact sparse target where it exists (real points), filling only the new pixels.
                pc_d = torch.tensor(cam_model.depth2pc(dense_meters), device=device)      # perturbed-frame cloud
                pc_d_gt = rotate_forward(pc_d, RT)                                         # GT frame
                uv_d, depth_d, _, _, VI_d = cam_model.project_withindex_pytorch(pc_d_gt, self.real_shape)
                uv_d = uv_d.t().int().contiguous()
                uv_dRT, depth_dRT, _, _, VI_dRT = cam_model.project_withindex_pytorch(pc_d, self.real_shape)
                uv_dRT = uv_dRT.t().int().contiguous()
                delta_Pd, idx_d = self.delta_1(uv_dRT, uv_d, VI_dRT, VI_d)
                iRT_d = torch.arange(VI_dRT[idx_d].shape[0], device=device) + 1
                uv_dRT_af = uv_dRT[idx_d[VI_dRT], :]; depth_dRT_af = depth_dRT[idx_d[VI_dRT]]
                iuv_d = torch.arange(VI_d[idx_d].shape[0], device=device) + 1
                uv_d_af = uv_d[idx_d[VI_d], :]; depth_d_af = depth_d[idx_d[VI_d]]
                _, iRT_deoccl_d = self.gen_depth_img(uv_dRT_af, depth_dRT_af, iRT_d, cam_params)
                iRT_fresh_d = self.fresh_indexes(iRT_deoccl_d, iRT_d)
                _, iuv_deoccl_d = self.gen_depth_img(uv_d_af, depth_d_af, iuv_d, cam_params)
                iuv_fresh_d = self.fresh_indexes(iuv_deoccl_d, iuv_d)
                mask_d = (iuv_fresh_d > 0) & (iRT_fresh_d > 0)
                project_delta_P_dense = self.delta_2(delta_Pd, uv_dRT_af, mask_d)
                sp_valid = (project_delta_P[0] != 0) | (project_delta_P[1] != 0)           # real-point support
                project_delta_P_dense[:, sp_valid] = project_delta_P[:, sp_valid]          # exact sparse wins
                project_delta_P = project_delta_P_dense

            ## downsample and crop
            depth_img_no_occlusion /= MAX_DEPTH
            depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)
            depth_img_no_occlusion_RT_training = torch.cat((depth_img_no_occlusion_RT_training, depth_img_no_occlusion), dim=0)
            rgb, depth_img_no_occlusion_RT_training, project_delta_P, depth_mask\
                = self.DownsampleCrop_M3ED_delta_mask(rgb, depth_img_no_occlusion_RT_training, project_delta_P, depth_mask, split, h=h, w=w)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)
            depth_mask_gt.append(depth_mask)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)
        depth_mask_gt = torch.stack(depth_mask_gt)
        

        return rgb_input, depth_input, flow_gt, depth_mask_gt

    def push_fuse_dense_flow(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train'):
        depth_input = []
        rgb_input = []
        flow_gt = []
        depth_mask_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv_ori, depth_ori, _, _, VI_indexes_ori = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv_ori = uv_ori.t().int().contiguous()

            pc_rotated = rotate_back(pc, RT)
            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()
            
            ## complete point cloud
            depth_img_no_occlusion_RT_training, _ = self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            depth_img_no_occlusion_RT_training_dense = sparse_to_dense(depth_img_no_occlusion_RT_training.cpu().detach().numpy(),
                fill_level=(self.partial_fill or 'full'))   # respect partial_light => dense flow on the plight support (one-variable exp)
            depth_img_no_occlusion_RT_training_dense = torch.tensor(depth_img_no_occlusion_RT_training_dense, device=device)
            pc_rotated = cam_model.depth2pc(depth_img_no_occlusion_RT_training_dense)
            pc_rotated = torch.tensor(pc_rotated, device=device)
            pc = rotate_forward(pc_rotated, RT)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            ## make depth_image for training
            depth_img_no_occlusion_RT_training /= MAX_DEPTH
            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)
            depth_img_no_occlusion_RT_training_dense /= MAX_DEPTH
            depth_img_no_occlusion_RT_training_dense = depth_img_no_occlusion_RT_training_dense.unsqueeze(0)
            depth_img_no_occlusion_RT_training = torch.cat((depth_img_no_occlusion_RT_training, depth_img_no_occlusion_RT_training_dense), dim=0)

            # make depth image for generating depth mask
            depth_img_no_occlusion, _ = \
                self.gen_depth_img(uv_ori, depth_ori, VI_indexes_ori[VI_indexes_ori], cam_params)
            depth_img_no_occlusion_GT = sparse_to_dense(depth_img_no_occlusion.cpu().detach().numpy())
            depth_img_no_occlusion_GT = torch.tensor(depth_img_no_occlusion_GT, device=device)
            event_mask = (rgb[0, :, :] > 0) + (rgb[1, :, :] > 0)
            depth_img_no_occlusion_GT_masked = depth_img_no_occlusion_GT * torch.tensor(event_mask, device=device)
            pc_masked = cam_model.depth2pc(depth_img_no_occlusion_GT_masked)
            pc_masked_rotated = rotate_back(torch.tensor(pc_masked, device=device), RT)
            uv_masked_RT, depth_masked_RT, _, _, VI_masked_indexes_RT = cam_model.project_withindex_pytorch(pc_masked_rotated, self.real_shape)
            uv_masked_RT = uv_masked_RT.t().int().contiguous()
            depth_img_no_occlusion_masked_RT, _ = self.gen_depth_img(uv_masked_RT, depth_masked_RT, VI_masked_indexes_RT[VI_masked_indexes_RT], cam_params)
            depth_img_no_occlusion_masked_RT = depth_img_no_occlusion_masked_RT.unsqueeze(0)
            depth_mask = depth_img_no_occlusion_masked_RT>0
            depth_mask = torch.cat((depth_mask, event_mask.unsqueeze(0)), dim=0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            depth_img_no_occlusion_GT /= MAX_DEPTH
            depth_img_no_occlusion_GT = depth_img_no_occlusion_GT.unsqueeze(0)
            depth_img_no_occlusion_RT_training = torch.cat((depth_img_no_occlusion_RT_training, depth_img_no_occlusion_GT), dim=0)
            rgb, depth_img_no_occlusion_RT_training, project_delta_P, depth_mask\
                = self.DownsampleCrop_M3ED_delta_mask(rgb, depth_img_no_occlusion_RT_training, project_delta_P, depth_mask, split, h=h, w=w)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)
            depth_mask_gt.append(depth_mask)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)
        depth_mask_gt = torch.stack(depth_mask_gt)
        

        return rgb_input, depth_input, flow_gt, depth_mask_gt

    def push_input(self, rgbs, pcs, T_errs, R_errs, device, MAX_DEPTH=10., h=600, w=960, split='train'):
        depth_input = []
        rgb_input = []
        x_list = []
        y_list = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            if T_errs is not None:
                R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(T_errs[idx].to(device))
                RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))
                pc_rotated = rotate_back(pc, RT)    # Nx4
            else:
                pc_rotated = pc

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, _ = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)

            depth_img_no_occlusion_RT_training /= MAX_DEPTH

            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)

            ## downsample and crop
            rgb, depth_img_no_occlusion_RT_training, x, y \
                = self.DownsampleCrop_input(rgb, depth_img_no_occlusion_RT_training, split, h=h, w=w)
            x_list.append(x)
            y_list.append(y)

            rgb_input.append(rgb)
            depth_input.append(depth_img_no_occlusion_RT_training)

        depth_input = torch.stack(depth_input)
        rgb_input = torch.stack(rgb_input)

        return rgb_input, depth_input, x_list, y_list
    
    def push_flow(self, rgbs, pcs, T_errs, R_errs, x_list, y_list, device, offsets=None):
        flow_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = mathutils.Matrix(np.matmul(np.asarray(T), np.asarray(R)))

            pc_rotated = rotate_back(pc, RT)    # Nx4

            if offsets is not None:
                R_offset = offsets[0][idx]
                T_offset = offsets[1][idx]
                RT_offset = to_rotation_matrix(R_offset, T_offset)
                pc = rotate_back(pc, RT_offset)

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv, depth, _, _, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            project_delta_P \
                = self.DownsampleCrop_flow(project_delta_P, x_list[idx], y_list[idx])

            flow_gt.append(project_delta_P)

        flow_gt = torch.stack(flow_gt)

        return flow_gt

    def push_depth_gt(self, rgbs, pcs, x_list, y_list, device, MAX_DEPTH=10., offsets=None, downsample=None):
        lidar_depths = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)

            if downsample is not None:
                self.real_shape = [int(rgb.shape[1] // downsample), int(rgb.shape[2] // downsample), rgb.shape[0]]
                x_list[idx] = int(x_list[idx] // downsample)
                y_list[idx] = int(y_list[idx] // downsample)
            else:
                self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            if offsets is not None:
                R_offset = offsets[0][idx]
                T_offset = offsets[1][idx]
                RT_offset = to_rotation_matrix(R_offset, T_offset)
                pc = rotate_back(pc, RT_offset)

            if downsample is not None:
                cam_params = self.calibs[idx] / downsample
            else:
                cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv_RT, depth_RT, _, _, VI_indexes_RT = cam_model.project_withindex_pytorch(pc, self.real_shape)
            uv_RT = uv_RT.t().int().contiguous()

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, _ = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)
            depth_img_no_occlusion_RT_training /= MAX_DEPTH
            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)


            ## downsample and crop
            depth_img_no_occlusion_RT_training \
                = self.DownsampleCrop_flow(depth_img_no_occlusion_RT_training, x_list[idx], y_list[idx], downsample=downsample)
            
            lidar_depths.append(depth_img_no_occlusion_RT_training)
        
        return torch.stack(lidar_depths)