import torch
import torch.nn as nn

from .base_conv_gru import *
from .ode_func import ODEFunc
from .diffeq_solver import DiffeqSolver
from .utils import create_convnet,Tracker

class CombineLatentEmbeddings(nn.Module):
    def __init__(self, channels):
        super(CombineLatentEmbeddings, self).__init__()
        # First 3D convolution layer
        self.conv1 = nn.Conv3d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # Second 3D convolution layer to reduce back to original channel size
        self.conv2 = nn.Conv3d(channels * 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, baseline, difference):
        # Concatenate along the channel dimension (dim=1)
        combined = torch.cat((baseline, difference), dim=1)  # [b, c*2, h, w, d]
        out = self.conv1(combined)  # [b, c*2, h, w, d]
        out = self.relu(out)
        out = self.conv2(out)  # [b, c, h, w, d]
        return out
    
class VidODE(nn.Module):
    
    def __init__(self, args, input_dim, device):
        super(VidODE, self).__init__()
        
        self.args = args
        self.device = device
        # tracker
        self.tracker = Tracker()
        self.input_dim=input_dim
        self.args.n_layers = args.n_layers
        self.args.n_downs = args.n_downs
        # self.args.run_backwards = False
        self.args.dec_diff = 'dopri5'
        self.args.flowmap = args.flowmap
        self.args.ode_rnn = args.ode_rnn 
        # self.args.ode_n_unit = args.ode_n_unit
        # self.args.adjoint = args.adjoint
        # initial function
        self.build_model()
        

    
    def build_model(self):
        if self.args.downsample_latent:
            # channels for encoder, ODE, init decoder
            init_dim = self.args.embedding_dim*4
            resize = 2 ** self.args.n_downs
            base_dim = init_dim*resize
            # input_size = (self.args.input_size // resize, self.args.input_size // resize)
            input_size=(64/2,64/2)
            # ode_dim = base_dim
            ode_dim = base_dim
            ##### Conv Encoder
            self.encoder = Encoder(input_dim=self.input_dim,
                                ch=init_dim,
                                n_downs=self.args.n_downs).to(self.device)
        else:
            input_size=(64/2,64/2)
            base_dim = self.args.embedding_dim
            ode_dim = base_dim
        
        print(f"Building models... base_dim:{base_dim}")
        

        
        ##### ODE Encoder
        if self.args.ode_rnn:
            ode_func_netE = create_convnet(n_inputs=ode_dim,
                                        n_outputs=base_dim,
                                        n_layers=self.args.n_layers,
                                        n_units=base_dim // 2).to(self.device)
            
            rec_ode_func = ODEFunc(input_dim=ode_dim,
                                latent_dim=base_dim,  # channels after encoder, & latent dimension
                                ode_func_net=ode_func_netE,
                                device=self.device).to(self.device)
            
            z0_diffeq_solver = DiffeqSolver(base_dim,
                                            ode_func=rec_ode_func,
                                            method="euler",
                                            latents=base_dim,
                                            odeint_rtol=1e-3,
                                            odeint_atol=1e-4,
                                            device=self.device)
        else:
            z0_diffeq_solver = None
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
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
                                                ode_rnn = self.args.ode_rnn).to(self.device)
        
        ##### ODE Decoder
        ode_func_netD = create_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=self.args.n_layers,
                                       n_units=base_dim // 2).to(self.device)
        # ode_func_netD = create_convnet(n_inputs=ode_dim,
        #                                n_outputs=base_dim,
        #                                n_layers=self.args.n_layers,
        #                                n_units=128).to(self.device)
        
        gen_ode_func = ODEFunc(input_dim=ode_dim,
                               latent_dim=base_dim,
                               ode_func_net=ode_func_netD,
                               device=self.device).to(self.device)
        
        self.diffeq_solver = DiffeqSolver(base_dim,
                                          gen_ode_func,
                                          self.args.dec_diff, base_dim,
                                          odeint_rtol=1e-3,
                                          odeint_atol=1e-4,
                                          device=self.device)
        
        ##### Conv Decoder
        # self.combination_layers = CombineLatentEmbeddings(channels=self.input_dim).to(self.device)
        if not self.args.flowmap :
            self.decoder = Decoder(input_dim=base_dim, output_dim=self.input_dim, n_ups=self.args.n_downs).to(self.device)
            if self.args.classification :
                self.classifier = nn.Conv3d(self.input_dim, self.args.vocab_size, 3, 1, 1)
                
        else:
            self.decoder = Decoder(input_dim=base_dim*2, output_dim=self.input_dim*2 + 3, n_ups=self.args.n_downs).to(self.device)
            if self.args.classification :
                self.classifier = nn.Conv3d(self.input_dim, self.args.vocab_size, 3, 1, 1)

    def test_gpu(self,loc):
        print(torch.cuda.get_device_name(0))
        print('Memory Usage at loc:',loc)
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')             
    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        
        truth = truth.to(self.device)
        truth_time_steps = truth_time_steps.to(self.device)
        mask = mask.to(self.device)
        out_mask = out_mask.to(self.device)
        time_steps_to_predict = time_steps_to_predict.to(self.device)
        
        resize = 2 ** self.args.n_downs
        b, t, c, d, h, w = truth.shape
        pred_t_len = len(time_steps_to_predict)
        
        # # ##### Skip connection forwarding
        skip_image=[]
        for batch_idx in range(0,b):
            pos = torch.where(mask[batch_idx,:,:]==1)[0][-1]
            # skip_image.append(truth[batch_idx:(batch_idx+1), pos, ...]) if self.args.mode == 'extrapolation' else skip_image.append(truth[batch_idx:(batch_idx+1), 0, ...])
            skip_image.append(truth[batch_idx:(batch_idx+1), 0, ...])
        skip_image = torch.cat(skip_image,dim=0)
        skip_conn_embed = skip_image 
        # skip_conn_embed = self.encoder(skip_image).view(b, -1, d//resize, h // resize, w // resize)
        
        ##### Conv encoding
        if self.args.downsample_latent:
            e_truth = self.encoder(truth.view(b * t, c, d, h, w)).view(b, t, -1, d//resize, h // resize, w // resize)
        else:
            e_truth = truth
        
        ##### ODE encoding
        first_point_mu, first_point_std = self.encoder_z0(input_tensor=e_truth, time_steps=truth_time_steps, mask=mask, tracker=self.tracker)
        
        # Sampling latent features
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1, 1)
        
        # ==================================================================================== #
        
        ##### ODE decoding
        first_point_enc = first_point_enc.squeeze(0)
        sol_y_all = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        if not self.args.flowmap :
            b1, t1, c1, d1, h1, w1  = sol_y_all.shape
            # if self.args.downsample_latent:
            #     pred_x_all = self.decoder(sol_y_all.view(b1 * t1, c1, d1, h1, w1)).view(b1, t1, -1, d1*resize, h1*resize, w1*resize)
            # else:
            #     pred_x_all = sol_y_all
            
            pred_x_all = self.decoder(sol_y_all.view(b1 * t1, c1, d1, h1, w1)).view(b1, t1, -1, d1*resize, h1*resize, w1*resize)
            # index_selected = (truth_time_steps[out_mask[:,:,0].bool()]*self.args.timepoints).long()
            index_selected = (truth_time_steps*self.args.timepoints).long()
            pred_x = pred_x_all[0:pred_x_all.shape[0],index_selected,:]
            pred_x = pred_x[out_mask.squeeze(-1).bool(),:]
            if b==1:
                pred_x= pred_x.unsqueeze(0)
            pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
            true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
        else:
            b, t, c, d, h, w  = sol_y_all.shape
            index_selected = (truth_time_steps*self.args.timepoints).long()
            # sol_y = sol_y_all[0:sol_y_all.shape[0],index_selected,:]
            sol_y = sol_y_all


            # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
            # pred_outputs = self.get_flowmaps_new(sol_out=sol_y, first_prev_embed=skip_conn_embed,mask = out_mask) # b, t, 6, h, w
            # temp_mask = mask.clone()
            # temp_mask[:]=1
            temp_mask = torch.ones(b,pred_t_len,1)
            pred_outputs = self.get_flowmaps_new(sol_out=sol_y, first_prev_embed=skip_conn_embed,mask = temp_mask) # b, t, 6, h, w
            pred_outputs = torch.cat(pred_outputs, dim=1)
            pred_flows, pred_intermediates, pred_masks = \
                pred_outputs[:, :, 0:3, ...],pred_outputs[:, :, 3:(3+self.input_dim), ...], torch.sigmoid(pred_outputs[:, :, (3+self.input_dim):, ...])

            ### Warping first frame by using optical flow
            # Declare grid for warping
            # grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
            # grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
            # grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]
            # grid = torch.zeros(b, d, h, w, 3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # # Populate the identity grid to maintain the original spatial coordinates
            # for i in range(d):
            #     for j in range(h):
            #         for k in range(w):
            #             grid[:, i, j, k, 0] = 2.0 * k / (w - 1) - 1  # Normalized x-coordinate
            #             grid[:, i, j, k, 1] = 2.0 * j / (h - 1) - 1  # Normalized y-coordinate
            #             grid[:, i, j, k, 2] = 2.0 * i / (d - 1) - 1  # Normalized z-coordinate
            grid_z = torch.linspace(-1.0, 1.0, d).view(1, d, 1, 1, 1).expand(b, -1, h, w, -1)
            grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w, 1).expand(b, d, h, -1, -1)
            grid_y = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1, 1).expand(b, d, -1, w, -1)
            grid = torch.cat([grid_x, grid_y,grid_z], 4).float().to(self.device)  # [b, d, h, w, 2]
            # # Warping
            # # last_frame = truth[:, -1, ...] if self.opt.extrap else truth[:, 0, ...]
            last_frame = skip_image.clone()

            if not self.args.residual:
                # warped_pred_x = self.get_warped_images_new(pred_flows=pred_flows, start_image=last_frame, grid=grid,residual = self.args.residual)
                # warped_pred_x = torch.cat(warped_pred_x, dim=1)  # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
                # pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
                # pred_x = pred_x[0:b,index_selected,:]
                # pred_x = pred_x[out_mask.squeeze(-1).bool(),...]
                # if b==1:
                #     pred_x= pred_x.unsqueeze(0)
                # pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
                # true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
                pred_x = torch.zeros_like(pred_intermediates)
                prev_frame = last_frame.clone()
                for i in range(0,pred_x.shape[1]):
                    pred_x[:,i,:] = pred_masks[:,i,:]*prev_frame + (1-pred_masks[:,i,:])*pred_intermediates[:,i,:] 
                    prev_frame = pred_x[:,i,:].clone()
                pred_x = pred_x[0:b,index_selected,:]              
                pred_x = pred_x[out_mask.squeeze(-1).bool(),...]
                if b==1:
                    pred_x= pred_x.unsqueeze(0)
                pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
                true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]  

            else:
                add_combine=True
                if add_combine:
                    pred_x = torch.zeros_like(pred_intermediates)
                    pred_x = pred_x[0:b,index_selected,:]

                    # truth = torch.cat([last_frame.unsqueeze(0),truth],dim=1)
                    for i in range(0,pred_x.shape[1]):
                        pred_x[:,i,:] = last_frame + torch.sum(pred_intermediates[:,0:(index_selected[i]+1),:],dim=1)

                    pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
                    true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
                    pred_x = pred_x[out_mask.squeeze(-1).bool(),...]
                    # true_intermediates = pred_intermediates.clone()
                    if b==1:
                        pred_x= pred_x.unsqueeze(0)
                else:
                    # true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]
                    pred_x = torch.zeros_like(pred_intermediates)
                    prev_frame = last_frame.clone()
                    for i in range(0,pred_x.shape[1]):
                        pred_x[:,i,:] = self.combination_layers(prev_frame,pred_intermediates[:,i,:]) 
                        prev_frame = pred_x[:,i,:].clone()
                    pred_x = pred_x[0:b,index_selected,:]              
                    pred_x = pred_x[out_mask.squeeze(-1).bool(),...]
                    if b==1:
                        pred_x= pred_x.unsqueeze(0)
                    pred_intermediates_new = pred_x[:, 1:, ...] - pred_x[:, :-1, ...]
                    true_intermediates_new = truth[:, 1:, ...] - truth[:, :-1, ...]                     
                    # true_intermediates[:,i,:] = truth[:,i+1,:]-truth[:,i,:]
                
                    # if i==0:
                    #     true_intermediates[:,i,:]=0
                    # else:
                    #     true_intermediates[:,i,:] = truth[:,i,:]-truth[:,i-1,:]

                # pred_x = pred_x[out_mask.squeeze(-1).bool(),:]
                # true_intermediates_new = true_intermediates[out_mask.squeeze(-1).bool(),:]
                # pred_intermediates_new = pred_intermediates[out_mask.squeeze(-1).bool(),:]
            # pred_x = pred_x[out_mask[out_mask.bool()].view(b,-1).bool(),:]
            # true_intermediates = true_intermediates[out_mask[out_mask.bool()].view(b,-1).bool(),:]
            # pred_intermediates = pred_intermediates[out_mask[out_mask.bool()].view(b,-1).bool(),:]

        if self.args.classification:
            pred_x = self.classifier(pred_x)
            
            # pred_x = pred_x.view(b, -1, c, d, h, w)
        # truth[0,]
        return pred_x,index_selected,true_intermediates_new.detach(),pred_intermediates_new
        # else:
        #     # not ready yet
        #     ##### Conv decoding
        #     sol_y_all = sol_y_all.contiguous().view(b, pred_t_len, -1, d//resize, h // resize, w // resize)
        #     # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
        #     pred_outputs = self.get_flowmaps(sol_out=sol_y_all, first_prev_embed=skip_conn_embed, mask=out_mask) # b, t, 6, h, w
        #     pred_outputs = torch.cat(pred_outputs, dim=1)
        #     pred_flows, pred_intermediates, pred_masks = \
        #         pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.input_dim, ...], torch.sigmoid(pred_outputs[:, :, 2+self.input_dim:, ...])

        #     ### Warping first frame by using argsical flow
        #     # Declare grid for warping
        #     grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
        #     grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
        #     grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]

        #     # Warping
        #     last_frame = truth[:, -1, ...] if self.args.extrap else truth[:, 0, ...]
        #     warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
        #     warped_pred_x = torch.cat(warped_pred_x, dim=1)  # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w

        #     pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
            
        #     pred_x = pred_x.view(b, -1, c, h, w)
            
        #     ### extra information
        #     # extra_info = {}
            
        #     # extra_info["argsical_flow"] = pred_flows
        #     # extra_info["warped_pred_x"] = warped_pred_x
        #     # extra_info["pred_intermediates"] = pred_intermediates
        #     # extra_info["pred_masks"] = pred_masks

            
        #     # # extra_info = {}
            # return pred_x, pred_x_all
    
    def get_mse(self, truth, pred_x, mask=None):
    
        b, _, c, h, w = truth.size()
        
        if mask is None:
            selected_time_len = truth.size(1)
            selected_truth = truth
        else:
            selected_time_len = int(mask[0].sum())
            selected_truth = truth[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        loss = torch.sum(torch.abs(pred_x - selected_truth)) / (b * selected_time_len * c * h * w)
        return loss
    
    
    def get_diff(self, data, mask=None):
        
        data_diff = data[:, 1:, ...] - data[:, :-1, ...]
        b, _, c, h, w = data_diff.size()
        selected_time_len = int(mask[0].sum())
        masked_data_diff = data_diff[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        
        return masked_data_diff

    
    def export_infos(self):
        infos = self.tracker.export_info()
        self.tracker.clean_info()
        return infos
    
    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, _, c, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        
        if mask.size(1) == sol_out.size(1):
            sol_out = sol_out[mask.squeeze(-1).byte()].view(b, pred_time_steps, c, h, w)
        
        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder(cur_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
    
        return pred_flows
        
    def get_flowmaps_new(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, _, c, d, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
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
    
    def get_warped_images_new(self, pred_flows, start_image, grid,residual):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid

        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, d, h, w = pred_flows.shape
        
        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]           # b, 3,d, h, w
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :, :] / ((d - 1.0) / 2.0), pred_flow[:, 1:2, :, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 2:3, :, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 4, 1)   # b, d, h, w, 3
            if residual:
                flow_grid = grid.clone()
            else:
                flow_grid = grid.clone() + pred_flow.clone()# b, d, h, w, 3
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border", align_corners = True)
            pred_x += [warped_x.unsqueeze(1)]           # b, 1, 3, h, w
            last_frame = warped_x.clone()
        
        return pred_x 

    

    def get_warped_images(self, pred_flows, start_image, grid):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid

        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, h, w = pred_flows.shape
        
        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]           # b, 2, h, w
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)   # b, h, w, 2
            flow_grid = grid.clone() + pred_flow.clone()# b, h, w, 2
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border")
            pred_x += [warped_x.unsqueeze(1)]           # b, 1, 3, h, w
            last_frame = warped_x.clone()
        
        return pred_x
    
    def compute_all_losses(self, batch_dict):
        
        batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)

        sol_y, index_selected,true_intermediates,pred_intermediates= self.get_reconstruction(
            time_steps_to_predict=batch_dict["tp_to_predict"],
            truth=batch_dict["observed_data"],
            truth_time_steps=batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            out_mask=batch_dict["mask_predicted_data"])
        
        # # batch-wise mean
        # loss = torch.mean(self.get_mse(truth=batch_dict["data_to_predict"],
        #                                pred_x=pred_x,
        #                                mask=batch_dict["mask_predicted_data"]))

        # if not self.args.extrap:
        #     init_image = batch_dict["observed_data"][:, 0, ...]
        # else:
        #     init_image = batch_dict["observed_data"][:, -1, ...]

        # data = torch.cat([init_image.unsqueeze(1), batch_dict["data_to_predict"]], dim=1)
        # data_diff = self.get_diff(data=data, mask=batch_dict["mask_predicted_data"])

        # loss = loss + torch.mean(self.get_mse(truth=data_diff, pred_x=extra_info["pred_intermediates"], mask=None))

        # results = {}
        # results["loss"] = torch.mean(loss)
        # results["pred_y"] = pred_x

        
        return sol_y,index_selected,true_intermediates,pred_intermediates
