import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from model.latent_rectified_flow.model_utils import (
    LeadEmbedding,
    LearnablePositionalEncoding,
    Conv_MLP,
    AdaLayerNorm,
    Transpose,
    GELU2,
    series_decomp,
)


class FullAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        condition_embd,  # condition dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, condition_embed, mask=None):
        B, T, C = x.size()
        B, T_E, _ = condition_embed.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(condition_embed)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(condition_embed)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
    ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln_cross_1 = AdaLayerNorm(n_embd)
        self.ln_cross_2 = AdaLayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.cross_attn_1 = CrossAttention(
            n_embd=n_embd,
            condition_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.cross_attn_2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, cond, mask=None, text_emb=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.cross_attn_1(self.ln_cross_1(x, timestep), cond, mask=mask)
        x = x + a

        if text_emb is not None:
            a, att = self.cross_attn_2(
                self.ln_cross_2(x, timestep), text_emb, mask=mask
            )
            x = x + a

        x = x + self.mlp(self.ln2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=4,
        block_activate="GELU",
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self, input, t, cond, padding_masks=None, text_emb=None, label_emb=None
    ):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](
                x,
                t,
                mask=padding_masks,
                label_emb=label_emb,
                cond=cond,
                text_emb=text_emb,
            )
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        n_feature=4,
        n_temporal=125,
        n_layer_enc=6,
        n_embd=512,
        n_heads=4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        max_len=4,
        use_text=False,
        **kwargs
    ):
        super().__init__()
        self.ecg_emb = Conv_MLP(n_feature, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feature, resid_pdrop=resid_pdrop)

        self.temporal_encoder = Encoder(
            n_layer_enc,
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
        )

        self.pos_enc = LearnablePositionalEncoding(
            d_model=n_embd, dropout=resid_pdrop, max_len=max_len
        )

        self.use_text = use_text

    def forward(self, x, t, padding_masks=None, cond=None, text_emb=None):
        x = x.transpose(1, 2)
        x = self.ecg_emb(x)
        x = self.pos_enc(x)
        
        # Handle conditional input
        if cond is not None:
            cond = cond.transpose(1, 2)
            cond = self.ecg_emb(cond)
            cond = self.pos_enc(cond)
        
        # Ensure time dimension matches sequence length
        if len(t.shape) == 2:
            timesteps = t
        else:
            timesteps = t.unsqueeze(-1).repeat(1, x.shape[1])
            
        x = self.temporal_encoder(
            x, timesteps, cond=cond, text_emb=text_emb, padding_masks=padding_masks
        )

        x = self.inverse(x)
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    pass
