import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from model.latent_rectified_flow.transformer import Transformer
from model.latent_rectified_flow.model_utils import GELU2


def extend(t: torch.Tensor, shape: torch.Size):
    """Extend time tensor t to match input dimensions shape."""
    return t.view(-1, *([1] * (len(shape) - 1)))


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        feature_size,
        temporal_size,
        n_layer_enc=3,
        d_model=None,
        n_heads=4,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        use_text=False,
        text_encoder_url=None,
        train_num_points=4,
        **kwargs,
    ):
        super().__init__()

        self.feature_size = feature_size
        self.temporal_size = temporal_size
        self.use_text = use_text
        self.train_num_points = train_num_points
        self.global_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone
        self.model = Transformer(
            n_feature=feature_size,
            n_temporal=temporal_size,
            n_layer_enc=n_layer_enc,
            n_embd=d_model,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=temporal_size,
            use_text=use_text,
            text_encoder_url=text_encoder_url,
            **kwargs,
        )

        if self.use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(768, 512),
                GELU2(),
                nn.Linear(512, d_model),
            )

    def get_text_embed(self, report):
        text_emb = self.text_proj(report)
        text_emb = text_emb.unsqueeze(1)
        return text_emb

    @staticmethod
    def add_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Linear interpolation between x0 ~ pi_0 and x1 (noise) ~ pi_1 to obtain x_t.
        x_t = t * x1 + (1 - t) * x0
        """
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        t = extend(t, x0.shape)
        xt = t * noise + (1 - t) * x0
        return xt, noise

    def output(self, x, t, padding_masks=None, cond=None, report=None):
        """Model forward pass"""
        text_emb = self.get_text_embed(report) if self.use_text and report is not None else None
        t = t.view(-1, 1).repeat(1, x.shape[1])  # expand time dimension to match input
        return self.model(x, t, padding_masks=padding_masks, cond=cond, text_emb=text_emb)

    def forward(self, x, target=None, padding_masks=None, cond=None, report=None):
        """Training with rectified flow loss across multiple random time points."""
        b = x.shape[0]
        device = x.device
        total_loss = 0.0

        for _ in range(self.train_num_points):
            t = torch.rand((b,), device=device)           # t in [0, 1]
            x0 = torch.randn_like(target)                 # reference distribution N(0, I)
            x1 = target                                   # target latent
            t_ = t.view(b, *([1] * (x1.ndim - 1)))        # broadcast to data shape
            xt = (1.0 - t_) * x0 + t_ * x1                # straight path
            drift = x1 - x0                               # ground-truth vector field
            pred_drift = self.output(xt, t, padding_masks, cond, report)
            total_loss = total_loss + F.mse_loss(pred_drift, drift)

        return total_loss / self.train_num_points

    @torch.no_grad()
    def sample(self, shape, num_steps=100, cond=None, report=None):
        """Backward sampling using simple Euler integration."""
        device = next(self.parameters()).device
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)[:-1]
        for t in tqdm(timesteps):
            t_batch = torch.full((shape[0],), t, device=device)
            pred = self.output(x, t_batch, cond=cond, report=report)
            x = x + dt * pred
        return x

    @torch.no_grad()
    def sample_shift(self, shape, num_steps=10, cond=None, report=None):
        """Conditional generation wrapper."""
        return self.sample(shape, num_steps, cond, report)