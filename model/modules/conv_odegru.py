import torch
import torch.nn as nn

from .base_conv_gru import *
from .ode_func import ODEFunc
from .diffeq_solver import DiffeqSolver
from .utils import create_convnet, Tracker

class Latent_embedding_ODE(nn.Module):
    """Latent embedding prediction model using Neural ODEs and flow-based approach.
    
    This model uses a neural ODE to learn continuous dynamics in the latent space
    and generates predictions using a flow-based approach. It can operate in both
    residual and non-residual modes.
    """
    
    def __init__(self, args, input_dim, device):
        """Initialize the Latent_embedding_ODE model.
        
        Args:
            args: Configuration arguments
            input_dim: Input dimension (number of channels)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super(Latent_embedding_ODE, self).__init__()
        
        self.args = args
        self.device = device
        self.tracker = Tracker()
        self.input_dim = input_dim
        self.args.n_layers = args.n_layers
        self.args.n_downs = args.n_downs
        self.args.dec_diff = 'dopri5'
        self.args.residual = args.residual
        self.args.ode_rnn = args.ode_rnn
        
        self.build_model()

    def build_model(self):
        """Build the model architecture including encoder, ODE solver, and decoder."""
        # Determine dimensions based on whether to downsample latent space
        if self.args.downsample_latent:
            init_dim = self.args.embedding_dim * 4
            resize = 2 ** self.args.n_downs
            base_dim = init_dim * resize
            # input_size = (64/2, 64/2, 64/2)  # (depth, height, width)
            ode_dim = base_dim
            
            # Build encoder for downsampled latent space
            self.encoder = Encoder(
                input_dim=self.input_dim,
                ch=init_dim,
                n_downs=self.args.n_downs
            ).to(self.device)
        else:
            # input_size = (64/2, 64/2, 64/2)  # (depth, height, width)
            base_dim = self.args.embedding_dim
            ode_dim = base_dim

        print(f"Building models... base_dim:{base_dim}")

        # Build ODE encoder
        if self.args.ode_rnn:
            # Create ODE function network for encoder
            ode_func_netE = create_convnet(
                n_inputs=ode_dim,
                n_outputs=base_dim,
                n_layers=self.args.n_layers,
                n_units=base_dim // 2
            ).to(self.device)
            
            # Create ODE function for encoder
            rec_ode_func = ODEFunc(
                input_dim=ode_dim,
                latent_dim=base_dim,
                ode_func_net=ode_func_netE,
                device=self.device
            ).to(self.device)
            
            # Create ODE solver for encoder
            z0_diffeq_solver = DiffeqSolver(
                base_dim,
                ode_func=rec_ode_func,
                method="euler",
                latents=base_dim,
                odeint_rtol=1e-3,
                odeint_atol=1e-4,
                device=self.device
            )
        else:
            z0_diffeq_solver = None

        # Build encoder for initial state
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(
            input_dim=base_dim,
            hidden_dim=base_dim,
            kernel_size=(3, 3, 3),
            num_layers=1,
            dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
            batch_first=True,
            bias=True,
            return_all_layers=True,
            z0_diffeq_solver=z0_diffeq_solver,
            run_backwards=self.args.run_backwards,
            ode_rnn=self.args.ode_rnn
        ).to(self.device)

        # Build ODE decoder
        # Create ODE function network for decoder
        ode_func_netD = create_convnet(
            n_inputs=ode_dim,
            n_outputs=base_dim,
            n_layers=self.args.n_layers,
            n_units=base_dim // 2
        ).to(self.device)
        
        # Create ODE function for decoder
        gen_ode_func = ODEFunc(
            input_dim=ode_dim,
            latent_dim=base_dim,
            ode_func_net=ode_func_netD,
            device=self.device
        ).to(self.device)
        
        # Create ODE solver for decoder
        self.diffeq_solver = DiffeqSolver(
            base_dim,
            gen_ode_func,
            self.args.dec_diff,
            base_dim,
            odeint_rtol=1e-3,
            odeint_atol=1e-4,
            device=self.device
        )

        # Build decoder
        if not self.args.residual:
            self.decoder = Decoder(
                input_dim=base_dim,
                output_dim=self.input_dim,
                n_ups=self.args.n_downs
            ).to(self.device)
        else:
            self.decoder = Decoder(
                input_dim=base_dim*2,
                output_dim=self.input_dim*2 + 3,
                n_ups=self.args.n_downs
            ).to(self.device)

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        """Generate latent embeddings using the model.
        
        Args:
            time_steps_to_predict: Target time points for prediction
            truth: Ground truth latent embeddings
            truth_time_steps: Time points for ground truth latent embeddings
            mask: Mask for observed latent embeddings
            out_mask: Mask for latent embeddings to predict
            
        Returns:
            pred_x: Predicted frames
            index_selected: Selected time indices
            true_intermediates: Ground truth frame differences
            pred_intermediates: Predicted frame differences
        """
        # Move inputs to device
        truth = truth.to(self.device)
        truth_time_steps = truth_time_steps.to(self.device)
        mask = mask.to(self.device)
        out_mask = out_mask.to(self.device)
        time_steps_to_predict = time_steps_to_predict.to(self.device)
        
        # Validate input dimensions
        if truth.dim() != 6:
            raise ValueError(f"Expected 6D input tensor (b, t, c, d, h, w), got {truth.dim()}D")
        
        # Get dimensions
        resize = 2 ** self.args.n_downs
        b, t, c, d, h, w = truth.shape
        pred_t_len = len(time_steps_to_predict)
        
        # Get skip connection embedding from first frame
        skip_image = []
        for batch_idx in range(0, b):
            skip_image.append(truth[batch_idx:(batch_idx+1), 0, ...])
        skip_image = torch.cat(skip_image, dim=0)
        skip_conn_embed = skip_image
        
        # Encode input frames
        if self.args.downsample_latent:
            e_truth = self.encoder(truth.view(b * t, c, d, h, w)).view(b, t, -1, d//resize, h // resize, w // resize)
        else:
            e_truth = truth
        
        # Get initial state using ODE encoder
        first_point_mu, first_point_std = self.encoder_z0(
            input_tensor=e_truth,
            time_steps=truth_time_steps,
            mask=mask,
            tracker=self.tracker
        )
        
        # Prepare initial state for ODE solver
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1, 1)
        first_point_enc = first_point_enc.squeeze(0)
        
        # Solve ODE to get latent states
        sol_y_all = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        
        # Generate predictions
        if not self.args.residual:
            # Non-residual prediction path
            b1, t1, c1, d1, h1, w1 = sol_y_all.shape
            pred_x_all = self.decoder(sol_y_all.view(b1 * t1, c1, d1, h1, w1)).view(b1, t1, -1, d1*resize, h1*resize, w1*resize)
            index_selected = (truth_time_steps*self.args.timepoints).long()
            pred_x = pred_x_all[0:pred_x_all.shape[0], index_selected, :]
            pred_x = pred_x[out_mask.squeeze(-1).bool(), :]
            if b == 1:
                pred_x = pred_x.unsqueeze(0)
            pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
            true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
        else:
            # Residual prediction path
            b, t, c, d, h, w = sol_y_all.shape
            index_selected = (truth_time_steps*self.args.timepoints).long()
            sol_y = sol_y_all
            
            # Generate flowmaps
            temp_mask = torch.ones(b, pred_t_len, 1)
            pred_outputs = self.get_flowmaps_new(sol_out=sol_y, first_prev_embed=skip_conn_embed, mask=temp_mask)
            pred_outputs = torch.cat(pred_outputs, dim=1)
            
            # Split outputs into flows, intermediates, and masks
            pred_flows, pred_intermediates, pred_masks = \
                pred_outputs[:, :, 0:3, ...], \
                pred_outputs[:, :, 3:(3+self.input_dim), ...], \
                torch.sigmoid(pred_outputs[:, :, (3+self.input_dim):, ...])
            
            # Generate predictions using residual approach
            pred_x = torch.zeros_like(pred_intermediates)
            pred_x = pred_x[0:b, index_selected, :]
            last_frame = skip_conn_embed
            
            # Accumulate flowmaps to generate predictions
            for i in range(0, pred_x.shape[1]):
                pred_x[:, i, :] = last_frame + torch.sum(pred_intermediates[:, 0:(index_selected[i]+1), :], dim=1)
            
            # Compute frame differences
            pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
            true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
            pred_x = pred_x[out_mask.squeeze(-1).bool(), ...]
            if b == 1:
                pred_x = pred_x.unsqueeze(0)
        
        return pred_x, index_selected, true_intermediates_new.detach(), pred_intermediates_new

    def get_flowmaps_new(self, sol_out, first_prev_embed, mask):
        """Generate flowmaps between consecutive latent embeddings.
        
        Args:
            sol_out: Latent states from ODE solver
            first_prev_embed: Initial latent embedding
            mask: Mask for latent embeddings to predict
            
        Returns:
            pred_flows: List of predicted flowmaps between consecutive latent embeddings
        """
        b, _, c, d, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = []
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        
        if mask.size(1) == sol_out.size(1):
            sol_out = sol_out[mask.squeeze(-1).bool()].view(b, pred_time_steps, c, d, h, w)
        
        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder(cur_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
    
        return pred_flows

    def compute_all_losses(self, batch_dict):
        """Compute losses for training.
        
        Args:
            batch_dict: Dictionary containing batch data
                
        Returns:
            sol_y: Predicted latent embeddings
            index_selected: Selected time indices
            true_intermediates: Ground truth latent embeddings differences
            pred_intermediates: Predicted latent embeddings differences
        """
        # Move inputs to device
        batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)

        # Get predictions
        sol_y, index_selected, true_intermediates, pred_intermediates = self.get_reconstruction(
            time_steps_to_predict=batch_dict["tp_to_predict"],
            truth=batch_dict["observed_data"],
            truth_time_steps=batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            out_mask=batch_dict["mask_predicted_data"]
        )
        
        return sol_y, index_selected, true_intermediates, pred_intermediates
