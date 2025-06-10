"""
Base Convolutional GRU implementation for 3D data processing.
This module provides the core components for a Convolutional GRU network that can process 3D data
with ODE-based temporal modeling capabilities.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from .utils import get_device  # This import was removed but is still needed

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PADDING = 1
DEFAULT_STRIDE = 1
EPSILON = 1e-6  # Small value for numerical stability

class ConvGRUCell(nn.Module):
    """
    Convolutional GRU Cell for 3D data processing.
    
    This cell implements a GRU-like architecture using 3D convolutions instead of fully connected layers.
    It maintains spatial information throughout the network while processing temporal dependencies.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 kernel_size: Tuple[int, int, int], 
                 bias: bool = True, 
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize ConvGRU Cell.
        
        Args:
            input_dim: Number of channels in input tensor
            hidden_dim: Number of channels in hidden state
            kernel_size: Size of the convolutional kernel
            bias: Whether to add bias to convolutions
            dtype: Data type for tensors (cuda or cpu)
        """
        super(ConvGRUCell, self).__init__()
        # self.depth, self.height, self.width = input_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        
        # Convolutional layers for gates and candidate memory
        self.conv_gates = nn.Conv3d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate and reset_gate
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        self.conv_can = nn.Conv3d(
            in_channels=input_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def forward(self, 
                input_tensor: torch.Tensor, 
                h_cur: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of ConvGRU cell.
        
        Args:
            input_tensor: Input tensor of shape (b, c, d, h, w)
            h_cur: Current hidden state of shape (b, c_hidden, d, h, w)
            mask: Optional mask tensor of shape (b, 1, 1, 1, 1)
            
        Returns:
            Next hidden state of shape (b, c_hidden, d, h, w)
        """
        # Input validation
        if input_tensor.dim() != 5:
            raise ValueError(f"Expected 5D input tensor, got {input_tensor.dim()}D")
        if h_cur.dim() != 5:
            raise ValueError(f"Expected 5D hidden state, got {h_cur.dim()}D")
            
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute gates
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        
        # Compute candidate memory
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        
        # Update hidden state
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(-1, 1, 1, 1, 1).expand_as(h_cur)
            h_next = mask * h_next + (1 - mask) * h_cur
        
        return h_next


class Encoder_z0_ODE_ConvGRU(nn.Module):
    """
    Encoder that combines ODE solving with ConvGRU for processing 3D temporal data.
    This encoder uses a ConvGRU to process spatial information and an ODE solver for temporal modeling.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dtype, 
                 batch_first=False, bias=True, return_all_layers=False, 
                 z0_diffeq_solver=None, run_backwards=None, ode_rnn=False):
        """
        Initialize the ODE-ConvGRU encoder.
        
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels (can be list for multiple layers)
            kernel_size: Size of convolutional kernels
            num_layers: Number of ConvGRU layers
            dtype: Data type for tensors
            batch_first: Whether batch dimension is first
            bias: Whether to use bias in convolutions
            return_all_layers: Whether to return all layer outputs
            z0_diffeq_solver: ODE solver module
            run_backwards: Whether to process sequence backwards
            ode_rnn: Whether to use ODE-RNN hybrid
        """
        super(Encoder_z0_ODE_ConvGRU, self).__init__()
        
        # Validate and extend parameters for multiple layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length for kernel_size and hidden_dim')
        
        # Store parameters
        # self.depth, self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.z0_diffeq_solver = z0_diffeq_solver
        self.run_backwards = run_backwards
        self.ode_rnn = ode_rnn
        
        # Initialize tracking dictionary
        self.by_product = {}
        
        # Create ConvGRU cells
        self.cell_list = nn.ModuleList([
            ConvGRUCell(
                # input_size=(self.depth, self.height, self.width),
                input_dim=input_dim if i == 0 else hidden_dim[i - 1],
                hidden_dim=hidden_dim[i],
                kernel_size=kernel_size[i],
                bias=bias,
                dtype=dtype
            ) for i in range(num_layers)
        ])
        
        # Initialize transformation layers
        self.z0_dim = hidden_dim[0]
        z = hidden_dim[0]
        self.transform_z0 = nn.Sequential(
            nn.Conv3d(z, z, 1, 1, 0),
            nn.ReLU(),
            nn.Conv3d(z, z * 2, 1, 1, 0)
        )
    
    def forward(self, input_tensor, time_steps, mask=None, tracker=None):
        """
        Forward pass through the encoder.
        
        Args:
            input_tensor: Input tensor of shape (b, t, c, d, h, w) or (t, b, c, d, h, w)
            time_steps: Time steps for ODE solving
            mask: Optional mask tensor
            tracker: Optional tracker for debugging
            
        Returns:
            mean_z0: Mean of initial state
            std_z0: Standard deviation of initial state
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)
            
        last_yi, latent_ys = self.run_ode_conv_gru(
            input_tensor=input_tensor,
            mask=mask,
            time_steps=time_steps,
            run_backwards=self.run_backwards,
            tracker=tracker
        )
        
        # Transform final state to get mean and std
        trans_last_yi = self.transform_z0(last_yi)
        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()
        
        return mean_z0, std_z0
    
    def run_ode_conv_gru(self, input_tensor, mask, time_steps, run_backwards=True, tracker=None):
        """
        Run ODE-ConvGRU processing.
        
        Args:
            input_tensor: Input tensor
            mask: Mask tensor
            time_steps: Time steps for ODE solving
            run_backwards: Whether to process backwards
            tracker: Optional tracker for debugging
            
        Returns:
            yi_allbatch: Final hidden states
            latent_ys: All intermediate states
        """
        b, t, c, d, h, w = input_tensor.size()
        device = get_device(input_tensor)
        latent_ys = []
        yi_allbatch = []
        
        for batch_idx in range(b):
            # Initialize hidden state
            prev_input_tensor = torch.zeros((1, c, d, h, w)).to(device)
            
            # Get valid time steps and mask for this batch
            pos = torch.where(mask[batch_idx,:,:]==1)[0][-1]+1
            batch_time_steps = time_steps[0:pos]
            batch_mask = mask[batch_idx:(batch_idx+1),0:pos,:]
            prev_t, t_i = batch_time_steps[-1] + 0.01, batch_time_steps[-1]
            
            # Process time steps
            time_points_iter = range(0, batch_time_steps.size(-1))
            if run_backwards:
                time_points_iter = reversed(time_points_iter)
            
            for idx, i in enumerate(time_points_iter):
                # Solve ODE step
                if self.ode_rnn:
                    inc = self.z0_diffeq_solver.ode_func(prev_t, prev_input_tensor) * (t_i - prev_t)
                    if torch.isnan(inc).any():
                        raise ValueError("NaN values in ODE solution")
                    if tracker:
                        tracker.write_info(key=f"inc{idx}", value=inc.clone().cpu())
                    ode_sol = prev_input_tensor + inc
                else:
                    ode_sol = prev_input_tensor
                
                if tracker:
                    tracker.write_info(key=f"prev_input_tensor{idx}", value=prev_input_tensor.clone().cpu())
                    tracker.write_info(key=f"ode_sol{idx}", value=ode_sol.clone().cpu())
                
                # Stack solutions
                ode_sol = torch.stack((prev_input_tensor, ode_sol), dim=1)
                if torch.isnan(ode_sol).any():
                    raise ValueError("NaN values in stacked ODE solution")
                
                # Validate ODE solution
                if torch.mean(ode_sol[:, 0, :] - prev_input_tensor) >= 0.001:
                    raise ValueError(
                        f"ODE solution error: first point differs from initial value by "
                        f"{torch.mean(ode_sol[:, 0, :] - prev_input_tensor)}"
                    )
                
                # Process through ConvGRU
                yi_ode = ode_sol[:, -1, :]
                xi = input_tensor[:, i, :]
                yi = self.cell_list[0](
                    input_tensor=xi,
                    h_cur=yi_ode,
                    mask=batch_mask[:, i]
                )
                
                # Update for next iteration
                prev_input_tensor = yi
                prev_t, t_i = batch_time_steps[i], batch_time_steps[i - 1]
            
            yi_allbatch.append(yi)
        
        yi_allbatch = torch.cat(yi_allbatch, 0)
        return yi_allbatch, latent_ys
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    
    def _extend_for_multilayer(self, param, num_layers):
        """Extend parameter for multiple layers with validation."""
        if not isinstance(param, list):
            param = [param] * num_layers
        if len(param) != num_layers:
            raise ValueError(f"Parameter length {len(param)} does not match num_layers {num_layers}")
        return param


def get_norm_layer(ch: int) -> nn.Module:
    """
    Get normalization layer for 3D convolutions.
    
    Args:
        ch: Number of channels
        
    Returns:
        BatchNorm3d layer
    """
    return nn.BatchNorm3d(ch)

class Encoder(nn.Module):
    """
    3D Convolutional Encoder for processing volumetric data.
    This encoder progressively reduces spatial dimensions while increasing channel dimensions.
    """
    
    def __init__(self, input_dim: int = 3, ch: int = 64, n_downs: int = 2) -> None:
        """
        Initialize the encoder.
        
        Args:
            input_dim: Number of input channels
            ch: Base number of channels
            n_downs: Number of downsampling layers
        """
        super(Encoder, self).__init__()
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv3d(input_dim, ch, 3, 1, 1),
            get_norm_layer(ch),
            nn.ReLU()
        )
        
        # Downsampling layers
        self.down_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch * (2**i), ch * (2**(i+1)), 4, 2, 1),
                get_norm_layer(ch * (2**(i+1))),
                nn.ReLU()
            ) for i in range(n_downs)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (b, c, d, h, w)
            
        Returns:
            Encoded features
        """
        x = self.initial(x)
        for layer in self.down_layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    """
    3D Convolutional Decoder for reconstructing volumetric data.
    This decoder progressively increases spatial dimensions while decreasing channel dimensions.
    """
    
    def __init__(self, input_dim: int = 256, output_dim: int = 3, n_ups: int = 2) -> None:
        """
        Initialize the decoder.
        
        Args:
            input_dim: Number of input channels
            output_dim: Number of output channels
            n_ups: Number of upsampling layers
        """
        super(Decoder, self).__init__()
        
        # Upsampling layers
        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(input_dim // (2**i), input_dim // (2**(i+1)), 3, 1, 1),
                get_norm_layer(input_dim // (2**(i+1))),
                nn.ReLU()
            ) for i in range(n_ups)
        ])
        
        # Final layer
        self.final = nn.Conv3d(input_dim // (2**n_ups), output_dim, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tensor of shape (b, c, d, h, w)
            
        Returns:
            Reconstructed output
        """
        for layer in self.up_layers:
            x = layer(x)
        return self.final(x)