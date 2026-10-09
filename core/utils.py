import os, socket, time
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torch.utils.data.dataloader import default_collate


def _unique_logdir():
    # Unique per RUN so co-located cluster jobs (same node, same second) don't race on the same
    # tensorboard dir (default SummaryWriter logdir = runs/<time>_<host> -> FileExistsError collision).
    tag = os.environ.get('SLURM_JOB_ID') or str(os.getpid())
    return os.path.join('runs', f"{time.strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}_{tag}")


class Logger:
    def __init__(self, model, scheduler, SUM_FREQ=100):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.SUM_FREQ = SUM_FREQ

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(logdir=_unique_logdir())

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(logdir=_unique_logdir())

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, nums, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.epochs * nums + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs * nums + 100, gamma=0.1)
    return optimizer, scheduler


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'event_frame'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['event_frame'])
    returns['point_cloud'] = point_clouds
    returns['event_frame'] = imgs
    return returns

def merge_inputs_rgb(queries):
    point_clouds = []
    imgs = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    return returns


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, scale=8, mode='bilinear'):
    new_size = (scale * flow.shape[2], scale * flow.shape[3])
    return scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

import matplotlib.pyplot as plt
def feature_visualizer(feature_maps, save_dir):
    feature_map_mean = np.mean(feature_maps, axis=0)
    plt.figure()
    plt.imshow(feature_map_mean)
    plt.savefig(f"{save_dir}.png")

def visualize_distinctiveness_map(feat1, feat2, save_dir, title="Feature Distinctiveness Map"):
    """
    Visualize a distinctiveness map using a heatmap.
    """
    feat1_norm = F.normalize(feat1, p=2, dim=1)
    feat2_norm = F.normalize(feat2, p=2, dim=1)
    similarity = (feat1_norm * feat2_norm).sum(dim=1).squeeze(0)  # (H, W)
    
    # Convert cosine similarity to distinctiveness: 1 - (similarity + 1) / 2 → range [0, 1]
    distinctiveness = (similarity + 1.0) / 2.0

    plt.figure(figsize=(6, 6))
    plt.imshow(distinctiveness.cpu().numpy(), cmap='coolwarm')
    plt.colorbar(label='Distinctiveness (Hardness of Matching)')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}.png")

import cv2
def edge_nms(image):
    """
    Apply edge-based non-maximum suppression (NMS) to an edge-detected image.
    image: Grayscale edge-detected image (e.g., gradient magnitude of an edge filter)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Compute gradients in x and y directions using Sobel filters
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Step 2: Compute gradient magnitude and direction (angle)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)  # convert to degrees

    # Step 3: Quantize direction to nearest 0, 45, 90, or 135 degrees
    quantized_directions = np.zeros(grad_direction.shape, dtype=np.int32)
    quantized_directions[(grad_direction >= -22.5) & (grad_direction < 22.5) | 
                        (grad_direction >= 157.5) | (grad_direction < -157.5)] = 0  # Horizontal
    quantized_directions[(grad_direction >= 22.5) & (grad_direction < 67.5) | 
                        (grad_direction >= -157.5) & (grad_direction < -112.5)] = 45  # Diagonal 45
    quantized_directions[(grad_direction >= 67.5) & (grad_direction < 112.5) | 
                        (grad_direction >= -112.5) & (grad_direction < -67.5)] = 90  # Vertical
    quantized_directions[(grad_direction >= 112.5) & (grad_direction < 157.5) | 
                        (grad_direction >= -67.5) & (grad_direction < -22.5)] = 135  # Diagonal 135

    # Step 4: Apply Non-Maximum Suppression
    suppressed = np.zeros(grad_magnitude.shape, dtype=np.float32)
    H, W = grad_magnitude.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            direction = quantized_directions[i, j]
            if direction == 0:  # Horizontal
                neighbors = (grad_magnitude[i, j-1], grad_magnitude[i, j+1])
            elif direction == 45:  # Diagonal 45
                neighbors = (grad_magnitude[i-1, j+1], grad_magnitude[i+1, j-1])
            elif direction == 90:  # Vertical
                neighbors = (grad_magnitude[i-1, j], grad_magnitude[i+1, j])
            elif direction == 135:  # Diagonal 135
                neighbors = (grad_magnitude[i-1, j-1], grad_magnitude[i+1, j+1])

            # Suppress non-maximum pixels
            if grad_magnitude[i, j] >= neighbors[0] and grad_magnitude[i, j] >= neighbors[1]:
                suppressed[i, j] = grad_magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed