"""
Latent ODEs for Irregularly-Sampled Time Series
Author: Yulia Rubanova

This module implements various encoder-decoder architectures for processing irregularly sampled
time series data using ODE-based neural networks. It includes:
- GRU units with uncertainty handling
- RNN-based encoders
- ODE-RNN hybrid encoders
- Simple decoders
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical, Normal
from typing import Tuple, Optional, List, Dict, Any
from contextlib import nullcontext
from .utils import *
from torch.nn.modules.rnn import LSTM, GRU
from .utils import get_device

# Constants for configuration
DEFAULT_HIDDEN_SIZE = 100
DEFAULT_GRU_UNITS = 100
MINIMUM_STEP_RATIO = 50
EPSILON = 1e-6

class GRU_unit(nn.Module):
    """
    Gated Recurrent Unit for processing sequential data.
    
    Args:
        latent_dim: Dimension of latent space
        input_dim: Dimension of input data
        update_gate: Optional custom update gate network
        reset_gate: Optional custom reset gate network
        new_state_net: Optional custom new state network
        n_units: Number of units in hidden layers
        device: Device to run computations on
    """
    def __init__(self, 
                 latent_dim: int, 
                 input_dim: int,
                 update_gate: Optional[nn.Module] = None,
                 reset_gate: Optional[nn.Module] = None,
                 new_state_net: Optional[nn.Module] = None,
                 n_units: int = DEFAULT_GRU_UNITS,
                 device: torch.device = torch.device("cpu")) -> None:
        super(GRU_unit, self).__init__()
        
        # Initialize networks with proper weight initialization
        self._init_networks(latent_dim, input_dim, n_units, 
                           update_gate, reset_gate, new_state_net)
        
    def _init_networks(self, latent_dim: int, input_dim: int, n_units: int,
                      update_gate: Optional[nn.Module], 
                      reset_gate: Optional[nn.Module],
                      new_state_net: Optional[nn.Module]) -> None:
        """Initialize the GRU networks with proper architecture."""
        # Update gate network
        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.update_gate)
        else:
            self.update_gate = update_gate

        # Reset gate network
        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        # New state network
        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2))
            init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net

    def forward(self, 
                y_mean: torch.Tensor, 
                y_std: torch.Tensor, 
                x: torch.Tensor, 
                masked_update: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GRU unit.
        
        Args:
            y_mean: Mean of previous state
            y_std: Standard deviation of previous state
            x: Input tensor
            masked_update: Whether to use masked update
            
        Returns:
            Tuple of (new state mean, new state std)
        """
        # Input validation
        if y_mean.dim() != 3:
            raise ValueError(f"Expected 3D y_mean tensor, got {y_mean.dim()}D")
        if y_std.dim() != 3:
            raise ValueError(f"Expected 3D y_std tensor, got {y_std.dim()}D")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D x tensor, got {x.dim()}D")

        # Concatenate inputs for gate computation
        y_concat = torch.cat([y_mean, y_std, x], -1)

        # Compute gates
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        
        # Compute new state
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        # Update state
        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        if torch.isnan(new_y).any():
            raise ValueError("NaN values detected in new_y")

        # Apply masked update if requested
        if masked_update:
            # Extract mask from input (assumes x contains both data and mask)
            n_data_dims = x.size(-1)//2
            mask = x[:, :, n_data_dims:]
            check_mask(x[:, :, :n_data_dims], mask)
            
            # Create binary mask
            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

            if torch.isnan(mask).any():
                raise ValueError("NaN values detected in mask")

            # Apply mask
            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

            if torch.isnan(new_y).any():
                raise ValueError(
                    "NaN values detected after masked update. " +
                    f"Debug info: mask: {mask}, y_mean: {y_mean}"
                )

        new_y_std = new_y_std.abs()
        return new_y, new_y_std



class Encoder_z0_RNN(nn.Module):
    """
    RNN-based encoder for initial latent state estimation.
    
    This encoder uses a GRU to process sequential data and estimate the initial
    latent state (z0) of the system.
    
    Args:
        latent_dim: Dimension of latent space
        input_dim: Dimension of input data
        lstm_output_size: Size of GRU output
        use_delta_t: Whether to include time differences
        device: Device to run computations on
    """
    def __init__(self, 
                 latent_dim: int, 
                 input_dim: int, 
                 lstm_output_size: int = 20,
                 use_delta_t: bool = True, 
                 device: torch.device = torch.device("cpu")) -> None:
        super(Encoder_z0_RNN, self).__init__()
        
        # Input validation
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
            
        self.gru_rnn_output_size = lstm_output_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.use_delta_t = use_delta_t

        # Initialize networks
        self._init_networks()
        
    def _init_networks(self) -> None:
        """Initialize the encoder networks."""
        # Network for transforming GRU output to z0
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.gru_rnn_output_size, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)

        # GRU for sequence processing
        input_dim = self.input_dim + (1 if self.use_delta_t else 0)
        self.gru_rnn = GRU(input_dim, self.gru_rnn_output_size).to(self.device)

    def forward(self, data, time_steps, run_backwards = True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)
        # t0: not used here
        n_traj = data.size(0)

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        data = data.permute(1,0,2) 

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = reverse(data)

        if self.use_delta_t:
            delta_t = time_steps[1:] - time_steps[:-1]
            if run_backwards:
                # we are going backwards in time with
                delta_t = reverse(delta_t)
            # append zero delta t in the end
            delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
            data = torch.cat((delta_t, data),-1)

        outputs, _ = self.gru_rnn(data.float())

        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)





class Encoder_z0_ODE_RNN(nn.Module):
    """
    ODE-RNN encoder for initial latent state estimation.
    
    This encoder combines ODE solving with RNN processing to estimate the initial
    latent state (z0) of the system. It can process data both forwards and backwards
    in time.
    
    Args:
        latent_dim: Dimension of latent space
        input_dim: Dimension of input data
        z0_diffeq_solver: ODE solver module
        z0_dim: Optional dimension for z0 (defaults to latent_dim)
        GRU_update: Optional custom GRU update module
        n_gru_units: Number of units in GRU
        device: Device to run computations on
    """
    def __init__(self, 
                 latent_dim: int, 
                 input_dim: int,
                 z0_diffeq_solver: Optional[nn.Module] = None,
                 z0_dim: Optional[int] = None,
                 GRU_update: Optional[nn.Module] = None,
                 n_gru_units: int = DEFAULT_GRU_UNITS,
                 device: torch.device = torch.device("cpu")) -> None:
        super(Encoder_z0_ODE_RNN, self).__init__()
        
        # Input validation
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
            
        self.z0_dim = z0_dim if z0_dim is not None else latent_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        
        # Initialize networks
        self._init_networks(GRU_update, n_gru_units, z0_diffeq_solver)

    def _init_networks(self, GRU_update: Optional[nn.Module], n_gru_units: int, z0_diffeq_solver: Optional[nn.Module]) -> None:
        """Initialize the encoder networks."""
        if GRU_update is None:
            self.GRU_update = GRU_unit(self.latent_dim, self.input_dim, 
                n_units = n_gru_units, 
                device=self.device).to(self.device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver

        self.transform_z0 = nn.Sequential(
           nn.Linear(self.latent_dim * 2, 100),
           nn.Tanh(),
           nn.Linear(100, self.z0_dim * 2),)
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0


    def run_odernn(self, 
                   data: torch.Tensor, 
                   time_steps: torch.Tensor,
                   run_backwards: bool = True, 
                   save_info: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        # Input validation
        if data.dim() != 3:
            raise ValueError(f"Expected 3D data tensor, got {data.dim()}D")
        if time_steps.dim() != 1:
            raise ValueError(f"Expected 1D time_steps tensor, got {time_steps.dim()}D")
            
        if torch.isnan(data).any():
            raise ValueError("NaN values detected in input data")
        if torch.isnan(time_steps).any():
            raise ValueError("NaN values detected in time steps")

        # Validate ODE solution
        def validate_ode_solution(ode_sol: torch.Tensor, prev_y: torch.Tensor) -> None:
            diff = torch.mean(ode_sol[:, :, 0, :] - prev_y)
            if diff >= 0.001:
                raise ValueError(
                    f"ODE solution error: first point differs from initial value by {diff}"
                )

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        #print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert(not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert(not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp).to(device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert(not torch.isnan(ode_sol).any())

            validate_ode_solution(ode_sol, prev_y)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std			
            prev_t, t_i = time_points[i],  time_points[i-1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
                     "yi": yi.detach(), "yi_std": yi_std.detach(), 
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info



class Decoder(nn.Module):
    """
    Simple decoder for reconstructing data from latent space.
    
    This decoder transforms latent representations back to the original data space.
    
    Args:
        latent_dim: Dimension of latent space
        input_dim: Dimension of input/output data
        hidden_dim: Dimension of hidden layer
        dropout: Dropout probability
    """
    def __init__(self, 
                 latent_dim: int, 
                 input_dim: int,
                 hidden_dim: int = 100,
                 dropout: float = 0.1) -> None:
        super(Decoder, self).__init__()
        
        # Input validation
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
            
        # Initialize decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        init_network_weights(self.decoder)

    def forward(self, data):
        return self.decoder(data)


