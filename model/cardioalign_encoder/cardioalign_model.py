import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math


class SelfAttention(nn.Module):

    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length,
                         self.n_heads, self.d_head)

        # (B, S, D) -> (B, S, D * 3) -> 3 * (B, S, D) NOTE: nn.Linear multiplys last dimension of any given vector
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (B, S, D) -> (B, S, H, D/H) -> (B, H, S, D/H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H, S, S)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)
        output = weight @ v

        # (B, H, S, D/H) -> (B, S, H, D/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, S, D)
        return output


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features, L)
        residue = x

        # x: (B, Features, L) -> x: (B, L, Features)
        x = x.transpose(-1, -2)

        # x: (B, L, Features) -> x: (B, Features, L)
        x = self.attention(x)
        x = x.transpose(-1, -2)

        x = x + residue
        return x


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=15, padding=7)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=15, padding=7)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, In_channels, L)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class TrendLayer(nn.Module):
    """
    Models the trend component of a time series.
    """
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)

        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.permute(0, 2, 1)
        return trend_vals


class SeasonalLayer(nn.Module):
    """
    Models the seasonal component of a time series.
    """
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1)
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)
        all_seas_vals = all_seas_vals.permute(0, 2, 1)

        return all_seas_vals


class LevelModel(nn.Module):
    """
    Models the level component of a time series.
    """
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals

class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(1, 128, kernel_size=15, padding=7),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                nn.Conv1d(128, 128, kernel_size=15, stride=2, padding=6),
                VAE_ResidualBlock(128, 256),
                VAE_ResidualBlock(256, 256),
                nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=6),
                VAE_ResidualBlock(256, 512),
                VAE_ResidualBlock(512, 512),
                nn.Conv1d(512, 512, kernel_size=15, stride=2, padding=6),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                nn.Conv1d(512, 512, kernel_size=17, stride=2, padding=7),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                nn.Conv1d(512, 512, kernel_size=17, stride=2, padding=7),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
                nn.Conv1d(512, 8, kernel_size=15, padding=7),
                nn.Conv1d(8, 8, kernel_size=1, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, C)
        # noise: (B, C_Out, L/8)
        # output: (B, C_Out, L/8)
        # NOTE: apply noise after encoding
        x = x.transpose(1, 2)
        # x: (B, L, C) -> (B, C, L)
        # x = x.transpose(1, 2)
        # x: (B, C, L) -> (B, 8, L/8)
        for module in self.blocks:
            if getattr(module, "stride", None) == (2,):
                # Padding(left, right)
                x = F.pad(x, (0, 1))
            x = module(x)
            # print("encoder",x.shape)

        # (B, 8, L/8) -> 2 x (B, 4, L/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z ~ N(0, 1) -> x ~ N(mean, variance)
        # x = mean + stdev * z
        # (B, 4, L/8) -> (B, 4, L/8)
        if noise is None:
            noise = torch.randn(stdev.shape, device=stdev.device)
        x = mean + stdev * noise

        # Scale the output by a constant (magic number)
        x = x * 0.18215

        return x, mean, log_variance


class VAE_Residual_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(4, 4, kernel_size=1, padding=0),
                nn.Conv1d(4, 512, kernel_size=15, padding=7),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=15, padding=7),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=15, padding=7),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=15, padding=7),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),

                nn.Upsample(scale_factor=2),
                nn.Conv1d(256, 256, kernel_size=15, padding=7),
                VAE_ResidualBlock(256, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),

                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=15, padding=7),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),

                nn.GroupNorm(32, 128),
                nn.SiLU(),
                nn.Conv1d(128, 1, kernel_size=15, padding=7),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L/8)

        # Avoid in-place to prevent autograd versioning issues when the caller reuses 'x' later
        x = x / 0.18215

        for module in self.blocks:
            x = module(x)
            # print("decoder",x.shape)

        # (B, 12, L) -> (B, L, 12)
        x = x.transpose(1, 2)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self, seq_len=1280, feat_dim=1, latent_dim=64, trend_poly=2, custom_seas=[(10, 128)]):
        super().__init__()

        # Part 1: The original decoder now acts as the residual model
        self.residual_decoder = VAE_Residual_Decoder()

        # Part 2: Bridge to create a single latent vector 'z' from the encoder's distributed output
        # Encoder has 5 downsampling layers with stride 2, so L_out = L / 32
        encoder_out_len = seq_len // 32
        # Encoder outputs 4 channels for the latent code
        encoder_flat_dim = 4 * encoder_out_len
        self.to_latent = nn.Linear(encoder_flat_dim, latent_dim)

        # Part 3: The decomposable component models from TimeVAE
        self.level_model = LevelModel(latent_dim, feat_dim, seq_len)

        self.trend_layer = None
        if trend_poly is not None and trend_poly > 0:
            self.trend_layer = TrendLayer(seq_len, feat_dim, latent_dim, trend_poly)

        self.seasonal_layer = None
        if custom_seas is not None and len(custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(seq_len, feat_dim, latent_dim, custom_seas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L/32), the latent code from VAE_Encoder

        # --- 1. Residual Part ---
        # The original VAE_Decoder architecture is used to model the complex residuals
        residuals = self.residual_decoder(x)

        # --- 2. Decomposable Part ---
        # First, create the single latent vector 'z' needed by the decomposable models
        batch_size = x.shape[0]
        # Make contiguous before flattening to avoid as_strided gradients on non-contiguous tensors
        x_flat = x.contiguous().view(batch_size, -1)
        z = self.to_latent(x_flat)

        # Start with the level component
        decomposed_output = self.level_model(z)

        # Add trend if configured
        if self.trend_layer:
            decomposed_output += self.trend_layer(z)

        # Add seasonal components if configured
        if self.seasonal_layer:
            decomposed_output += self.seasonal_layer(z)

        # --- 3. Combine ---
        # The final output is the sum of the decomposable parts and the residual part
        final_output = decomposed_output + residuals

        return final_output


def loss_function(recons, x, mu, log_var, kld_weight=1e-4, use_fft_loss=False, fft_weight=1.0) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :para recons: reconstruction vector
    :para x: original vector
    :para mu: mean of latent gaussian distribution
    :log_var: log of latent gaussian distribution variance
    :kld_weight: weight of kl-divergence term
    :use_fft_loss: whether to include FFT loss
    :fft_weight: weight of fft_loss term
    """
    # recons, x: (B, L, 12) -> number, batch wise average
    # recons_loss = F.mse_loss(recons, x, reduction='sum').div(x.size(0))
    recons_loss = F.mse_loss(recons, x, reduction="mean")

    # (old) mu, log_var: (B, 4, L/8) -> number
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2), dim=1).sum()

    # q(z|x): distribution learned by encoder
    q_z_x = Normal(mu, log_var.mul(0.5).exp())
    # p(z): prior of z, intended to be standard Gaussian
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    # kld_loss: batch wise average
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()
    loss = recons_loss + kld_weight * kld_loss
    # loss = recons_loss

    fourier_loss = torch.tensor(0.0, device=loss.device)
    if use_fft_loss:
        # print("该模型使用了fft算法！")
        fft1 = torch.fft.fft(recons.transpose(1, 2), norm="forward")
        fft2 = torch.fft.fft(x.transpose(1, 2), norm="forward")
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = F.mse_loss(
            torch.real(fft1), torch.real(fft2), reduction="mean"
        ) + F.mse_loss(torch.imag(fft1), torch.imag(fft2), reduction="mean")
        loss += fft_weight * fourier_loss

    return_dict = {"loss": loss, "mse": recons_loss.detach(), "KLD": kld_loss.detach()}
    if use_fft_loss:
        return_dict["fft_loss"] = fourier_loss.detach()

    return return_dict


if __name__ == "__main__":
    
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    data = torch.randn(64, 1280, 1)

    latent_code, mean, log_var = encoder(data)
    print(f"Input data shape: {data.shape}")
    print(f"Latent code shape: {latent_code.shape}")

    reconstruction = decoder(latent_code)
    print(f"Reconstruction shape: {reconstruction.shape}")

    loss = loss_function(reconstruction, data, mean, log_var)
    print(f"Computed Loss: {loss}")

    assert data.shape == reconstruction.shape
    print("\nModel test with decompositional decoder passed successfully!")

