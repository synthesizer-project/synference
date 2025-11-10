
from synference import SBI_Fitter
from unyt import um
from ili import FCN
import numpy as np
from astropy.io import fits

import torch
from torch import nn
from torch.nn import functional as F

class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss Function.
    This loss is smoother than L2 loss and less sensitive to outliers.
    It is twice differentiable everywhere, unlike Huber loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # The `torch.log(torch.cosh(x))` is numerically stable enough for typical network outputs.
        diff = y_pred - y_true
        return torch.log(torch.cosh(diff)).sum()

class SpenderLikeVAE(nn.Module):
    """
    A VAE with encoder architecture matching SPENDER.

    The encoder consists of:
    1. A 3-layer 1D CNN with increasing kernel sizes, instance normalization, 
       PReLU activation, dropout, and max pooling.
    2. A dot-product attention mechanism over the wavelength dimension.
    3. A 3-layer MLP to produce the parameters of the latent distribution.

    The decoder is a simple 3-layer MLP that reconstructs the spectrum from a
    latent sample.
    """
    def __init__(self, input_dim=1001, latent_dim=16, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --- ENCODER (SPENDER-style) ---
        # 1. Convolutional Frontend with normalization and dropout
        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        
        # Build conv blocks matching SPENDER
        self.conv_blocks = nn.ModuleList()
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i - 1]
            f_out = filters[i]
            kernel_size = sizes[i]
            padding = kernel_size // 2
            
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=f_in, out_channels=f_out, 
                         kernel_size=kernel_size, padding=padding),
                nn.InstanceNorm1d(f_out),
                nn.PReLU(f_out),
                nn.Dropout(p=dropout)
            )
            self.conv_blocks.append(conv_block)
        
        # Pooling layers (match SPENDER)
        self.pool1 = nn.MaxPool1d(sizes[0], padding=sizes[0] // 2)
        self.pool2 = nn.MaxPool1d(sizes[1], padding=sizes[1] // 2)
        
        # Calculate the length after pooling
        conv_output_len = input_dim
        # After pool1
        conv_output_len = (conv_output_len + 2 * (sizes[0] // 2)) // sizes[0]
        # After pool2
        conv_output_len = (conv_output_len + 2 * (sizes[1] // 2)) // sizes[1]
        
        # 2. Attention Module
        # Split channels into h and k for attention
        self.n_feature = filters[-1] // 2
        
        # 3. MLP for latent space parameters (match SPENDER structure)
        n_hidden = [128, 64, 32]
        
        self.enc_fc1 = nn.Linear(self.n_feature, n_hidden[0])
        self.enc_act1 = nn.PReLU(n_hidden[0])
        self.enc_drop1 = nn.Dropout(p=dropout)
        
        self.enc_fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.enc_act2 = nn.PReLU(n_hidden[1])
        self.enc_drop2 = nn.Dropout(p=dropout)
        
        self.enc_fc3 = nn.Linear(n_hidden[1], n_hidden[2])
        self.enc_act3 = nn.PReLU(n_hidden[2])
        self.enc_drop3 = nn.Dropout(p=dropout)
        
        # Output layers for mu and log_var
        self.enc_fc_mu = nn.Linear(n_hidden[2], latent_dim)
        self.enc_fc_log_var = nn.Linear(n_hidden[2], latent_dim)

        # --- DECODER ---
        # A simple 3-layer MLP as described in the spender paper
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_fc3 = nn.Linear(256, input_dim)

        # Decoder activation
        self.act_fn = nn.SiLU()

    def attention(self, x):
        """
        Dot-product attention mechanism as described in the SPENDER paper.
        Splits channels into two halves (h and k), computes softmax attention
        weights from k, and applies them to h.
        """
        # x shape: (batch, channels, length)
        # Split channels into two halves
        h = x[:, :self.n_feature, :]
        k = x[:, self.n_feature:, :]

        # Softmax operates on the wavelength dimension (last dimension)
        softmax_k = F.softmax(k, dim=-1)

        # Element-wise product
        e = h * softmax_k
        
        # Sum across wavelength dimension (like SPENDER)
        e = torch.sum(e, dim=2)
        
        return e

    def encode(self, x):
        """
        Encode input spectrum to latent distribution parameters.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Input spectra
            
        Returns
        -------
        mu : torch.Tensor, shape (batch, latent_dim)
            Mean of latent distribution
        log_var : torch.Tensor, shape (batch, latent_dim)
            Log variance of latent distribution
        """
        # Reshape input for Conv1d: (batch, channels, length)
        x = x.unsqueeze(1)

        # CNN frontend with pooling (SPENDER-style)
        x = self.pool1(self.conv_blocks[0](x))
        x = self.pool2(self.conv_blocks[1](x))
        x = self.conv_blocks[2](x)

        # Attention (reduces wavelength dimension)
        x = self.attention(x)

        # MLP with PReLU activations and dropout
        x = self.enc_drop1(self.enc_act1(self.enc_fc1(x)))
        x = self.enc_drop2(self.enc_act2(self.enc_fc2(x)))
        x = self.enc_drop3(self.enc_act3(self.enc_fc3(x)))
        
        # Output mu and log_var
        mu = self.enc_fc_mu(x)
        log_var = self.enc_fc_log_var(x)
        
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution
        log_var : torch.Tensor
            Log variance of latent distribution
            
        Returns
        -------
        z : torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vector to reconstructed spectrum.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch, latent_dim)
            Latent vectors
            
        Returns
        -------
        reconstructed_x : torch.Tensor, shape (batch, input_dim)
            Reconstructed spectra
        """
        x = self.act_fn(self.dec_fc1(z))
        x = self.act_fn(self.dec_fc2(x))
        # No final activation to allow for arbitrary flux values
        reconstructed_x = self.dec_fc3(x)
        return reconstructed_x

    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Input spectra
            
        Returns
        -------
        reconstructed_x : torch.Tensor, shape (batch, input_dim)
            Reconstructed spectra
        mu : torch.Tensor, shape (batch, latent_dim)
            Mean of latent distribution
        log_var : torch.Tensor, shape (batch, latent_dim)
            Log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        return mu
    
    def forward_vae(self, x):
        """
        Full VAE forward pass for training (regardless of embedding_mode).
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Input spectra
            
        Returns
        -------
        reconstructed_x : torch.Tensor, shape (batch, input_dim)
            Reconstructed spectra
        mu : torch.Tensor, shape (batch, latent_dim)
            Mean of latent distribution
        log_var : torch.Tensor, shape (batch, latent_dim)
            Log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    @property
    def n_parameters(self):
        """Number of trainable parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def vae_loss(recons_x, x, mu, log_var):
    """
    Calculates the VAE loss using the specified LogCoshLoss for reconstruction.
    
    Parameters
    ----------
    recons_x : torch.Tensor
        Reconstructed spectra
    x : torch.Tensor
        Original spectra
    mu : torch.Tensor
        Mean of latent distribution
    log_var : torch.Tensor
        Log variance of latent distribution
        
    Returns
    -------
    loss : torch.Tensor
        Total VAE loss (reconstruction + KL divergence)
    """
    log_cosh_loss = LogCoshLoss()
    recon_loss = log_cosh_loss(recons_x, x)

    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kld_loss


if __name__ == "__main__":

    fitter = SBI_Fitter.init_from_hdf5(
        "Spectra_BPASS_Chab_Continuity_SFH_0.01_z_14_logN_4.4_Calzetti_v4_multinode",
        "/cosma7/data/dp276/dc-harv3/work/sbi_grids/grid_spectra_BPASS_Chab_Continuity_SFH_0.01_z_14_logN_4.4_Calzetti_v4_multinode.hdf5" # noqa: E501
    )
    tab = fits.getdata('/cosma/apps/dp276/dc-harv3/synference/priv/jwst_nirspec_prism_disp.fits')
    wavs = tab['WAVELENGTH'] * um
    R = tab['R']
    
    embedding_net = SpenderLikeVAE(input_dim=1000, latent_dim=64,  dropout=0.1)


    fitter.create_feature_array(flux_units="log10 nJy", crop_wavelength_range=(0.6, 5.0), 
                                resample_wavelengths=wavs, inst_resolution_wavelengths=wavs, 
                                inst_resolution_r=R, theory_r=np.inf, min_flux_value=-10)

    embedding_net = SpenderLikeVAE(input_dim=len(fitter.feature_array[0]), latent_dim=64,  dropout=0.1)

    # n_hidden = [256, 128, 64]
    # embedding_net= FCN(n_hidden=n_hidden, n_input=len(fitter.feature_array[0]))



    fitter.run_single_sbi(
        model_type='nsf',
        num_transforms=35,
        learning_rate=5e-4,
        training_batch_size=256,
        embedding_net=embedding_net,
        name_append='spender_vae_latent64'
    )