# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import math
import numpy as np
import sys
import imageio
from torch.nn.modules.batchnorm import _BatchNorm

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    """
    Shifts tensor dimensions from source to destination position.
    Example: shift_dim(x, 1, -1) transforms (b, c, t, h, w) -> (b, t, h, w, c)
    
    Args:
        x (torch.Tensor): Input tensor
        src_dim (int): Source dimension index
        dest_dim (int): Destination dimension index
        make_contiguous (bool): Whether to make the output tensor contiguous
    
    Returns:
        torch.Tensor: Tensor with shifted dimensions
    """
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output (torch.Tensor): Model output tensor
        target (torch.Tensor): Ground truth tensor
        topk (tuple): Tuple of k values to compute accuracy for
    
    Returns:
        list: List of accuracy values for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adopt_weight(global_step, threshold=0, value=0.):
    """
    Adopts a weight value based on the global step and threshold.
    
    Args:
        global_step (int): Current training step
        threshold (int): Step threshold
        value (float): Weight value to use before threshold
    
    Returns:
        float: Weight value
    """
    weight = 1
    if global_step < threshold:
        weight = value
    return weight

def comp_getattr(args, attr_name, default=None):
    """
    Gets an attribute from args object with a default value if not found.
    
    Args:
        args: Object to get attribute from
        attr_name (str): Name of the attribute
        default: Default value if attribute not found
    
    Returns:
        Attribute value or default
    """
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default

def get_device(tensor):
    """
    Gets the device of a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor
    
    Returns:
        torch.device: Device of the tensor
    """
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def init_network_weights(net, std=0.1, mode=None):
    """
    Initializes network weights using specified initialization method.
    
    Args:
        net (nn.Module): Neural network to initialize
        std (float): Standard deviation for normal initialization
        mode (str): Initialization mode ('Cox' or None)
    """
    for idx, m in enumerate(net.modules()):
        if isinstance(m, nn.Linear):
            if mode == 'Cox':
                nn.init.normal_(m.weight, mean=0, std=std)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

def disable_running_stats(model):
    """
    Disables running statistics in batch normalization layers.
    
    Args:
        model (nn.Module): Model containing batch normalization layers
    """
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    """
    Re-enables running statistics in batch normalization layers.
    
    Args:
        model (nn.Module): Model containing batch normalization layers
    """
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def save_video_grid(video, fname, nrow=None, fps=6):
    """
    Saves a grid of videos to a file.
    
    Args:
        video (torch.Tensor): Video tensor of shape (b, c, t, h, w)
        fname (str): Output filename
        nrow (int, optional): Number of videos per row
        fps (int): Frames per second
    """
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)

def create_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, nonlinear='tanh'):
    """
    Creates a convolutional neural network.
    
    Args:
        n_inputs (int): Number of input channels
        n_outputs (int): Number of output channels
        n_layers (int): Number of convolutional layers
        n_units (int): Number of units in each layer
        nonlinear (str): Nonlinearity to use ('tanh' only supported)
    
    Returns:
        nn.Sequential: Convolutional neural network
    """
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    else:
        raise NotImplementedError('Only tanh nonlinearity is supported')
    
    layers = []
    layers.append(nn.Conv3d(n_inputs, n_units, 3, 1, 1, dilation=1))
    
    for i in range(n_layers):
        layers.append(nonlinear)
        layers.append(nn.Conv3d(n_units, n_units, 3, 1, 1, dilation=1))
    
    layers.append(nonlinear)
    layers.append(nn.Conv3d(n_units, n_outputs, 3, 1, 1, dilation=1))
    
    return nn.Sequential(*layers)

def reverse(tensor):
    """
    Reverses a tensor along its first dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor
    
    Returns:
        torch.Tensor: Reversed tensor
    """
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    return tensor[idx]

def linspace_vector(start, end, n_points):
    """
    Creates a vector of linearly spaced points between start and end.
    
    Args:
        start (torch.Tensor): Start points
        end (torch.Tensor): End points
        n_points (int): Number of points to generate
    
    Returns:
        torch.Tensor: Linearly spaced points
    """
    size = np.prod(start.size())
    assert(start.size() == end.size())
    if size == 1:
        res = torch.linspace(start, end, n_points)
    else:
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, 
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def check_mask(data, mask):
    """
    Checks that mask argument contains a valid mask for data.
    
    Args:
        data (torch.Tensor): Input data
        mask (torch.Tensor): Mask to check
    """
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()
    assert((n_zeros + n_ones) == np.prod(list(mask.size())))
    assert(torch.sum(data[mask == 0.] != 0.) == 0)

class Tracker:
    """
    Utility class for tracking information during model execution.
    """
    def __init__(self):
        self.infos = {}
    
    def write_info(self, key, value):
        """
        Writes information to the tracker.
        
        Args:
            key (str): Information key
            value: Information value
        """
        self.infos[key] = value
    
    def export_info(self):
        """
        Exports all tracked information.
        
        Returns:
            dict: Dictionary of tracked information
        """
        return self.infos
    
    def clean_info(self):
        """
        Clears all tracked information.
        """
        self.infos = {}


