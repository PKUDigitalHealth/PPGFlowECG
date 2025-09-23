import sys
import torch
import argparse
import logging
import os
import gc
import numpy as np
import torch.nn.functional as F


project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from model.cardioalign_encoder.cardioalign_model import VAE_Decoder, VAE_Encoder, loss_function
from utils.io_utils import load_yaml_config, seed_everything
from utils.ppgecg_dataset import ECGPPGLMDBDataset
from torch.utils.data import Dataset, DataLoader
from utils.data import ECGDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/cardioalign_encoder.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/cardioalign_encoder/",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    return parser.parse_args()

def _kl_gaussians(mu_p: torch.Tensor, logvar_p: torch.Tensor, mu_q: torch.Tensor, logvar_q: torch.Tensor) -> torch.Tensor:
    """KL(N_p || N_q) for diagonal Gaussians."""
    var_p = logvar_p.exp()
    var_q = logvar_q.exp()
    kl = 0.5 * (
        (var_p / var_q).sum(dim=-1)
        + ((mu_q - mu_p) ** 2 / var_q).sum(dim=-1)
        - mu_p.shape[-1]
        + (logvar_q - logvar_p).sum(dim=-1)
    )
    return kl.mean()

def _infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Simple InfoNCE with cosine similarity across batch.
    Accepts [B, D] or sequence latents [B, T, D...] and pools to [B, D].
    """
    def _pool_to_bf(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() >= 3:
            reduce_dims = tuple(range(1, x.dim() - 1))
            if len(reduce_dims) > 0:
                x = x.mean(dim=reduce_dims)
            return x
        return x.unsqueeze(-1)

    z1 = _pool_to_bf(z1)
    z2 = _pool_to_bf(z2)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature  # [B, B]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)

def train_loop(
    dataloader,
    encoder_ecg,
    encoder_ppg,
    decoder_ecg,
    decoder_ppg,
    loss_fn,
    optimizer,
    scheduler,
    device,
    kld_weight=1e-4,
    lambda_align=1e-2,
    lambda_cross=0.0,
    lambda_infonce=0.0,
    use_fft_loss=False,
    fft_weight=1.0,
    save_weights_path=None,
    logger=None,
    total_iterations=100000,
    save_interval=5000,
    log_interval=100,
):
    size = len(dataloader.dataset)
    encoder_ecg.train()
    if encoder_ppg is not None:
        encoder_ppg.train()
    decoder_ecg.train()
    decoder_ppg.train()

    iteration = 0
    dataloader_iterator = iter(dataloader)

    while iteration < total_iterations:
        try:
            file, ppg, ecg = next(dataloader_iterator)
            # ppg, ecg = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            file, ppg, ecg = next(dataloader_iterator)
            # ppg, ecg = next(dataloader_iterator)

        iteration += 1
        torch.autograd.set_detect_anomaly(True)
        ppg = ppg.unsqueeze(-1).to(device).float()
        ecg = ecg.unsqueeze(-1).to(device).float()

        # Encode
        z_ecg, mu_ecg, logvar_ecg = encoder_ecg(ecg)
        if encoder_ppg is None:
            # shared encoder
            z_ppg, mu_ppg, logvar_ppg = encoder_ecg(ppg)
        else:
            z_ppg, mu_ppg, logvar_ppg = encoder_ppg(ppg)

        # Decode (self-reconstruction)
        recons_ecg = decoder_ecg(z_ecg)
        recons_ppg = decoder_ppg(z_ppg)

        # Base VAE losses (optionally include FFT term)
        loss_dict_ecg = loss_fn(
            recons_ecg, ecg, mu_ecg, logvar_ecg, kld_weight, use_fft_loss, fft_weight
        )
        loss_dict_ppg = loss_fn(
            recons_ppg, ppg, mu_ppg, logvar_ppg, kld_weight, use_fft_loss, fft_weight
        )

        # Latent alignment losses
        align_l2 = F.mse_loss(mu_ecg, mu_ppg)

        # Symmetric KL between posterior Gaussians
        align_kl = 0.5 * (
            _kl_gaussians(mu_ecg, logvar_ecg, mu_ppg, logvar_ppg)
            + _kl_gaussians(mu_ppg, logvar_ppg, mu_ecg, logvar_ecg)
        )
        latent_align_loss = align_l2 + align_kl

        # Optional InfoNCE on latent samples
        infonce_loss = torch.tensor(0.0, device=ecg.device)
        if lambda_infonce > 0.0:
            infonce_loss = _infonce_loss(mu_ecg, mu_ppg)

        # Optional cross reconstruction
        cross_loss = torch.tensor(0.0, device=ecg.device)
        if lambda_cross > 0.0:
            cross_ecg_from_ppg = decoder_ecg(z_ppg.detach())
            cross_ppg_from_ecg = decoder_ppg(z_ecg.detach())
            # Only MSE on cross terms to be conservative
            cross_loss = F.mse_loss(cross_ecg_from_ppg, ecg) + F.mse_loss(cross_ppg_from_ecg, ppg)

        total_loss = (
            loss_dict_ecg["loss"]
            + loss_dict_ppg["loss"]
            + lambda_align * latent_align_loss
            + lambda_cross * cross_loss
            + lambda_infonce * infonce_loss
        )

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        if iteration % log_interval == 0:
            logger.info(
                f"Iteration {iteration}/{total_iterations} - "
                f"total: {total_loss.item():>7f} | "
                # f"ecg_loss: {loss_dict_ecg['loss'].item():>7f} (mse {loss_dict_ecg['mse'].item():>7f} fft {loss_dict_ecg['fft_loss'].item():>7f} KLD {loss_dict_ecg['KLD'].item():>7f}) | "
                # f"ppg_loss: {loss_dict_ppg['loss'].item():>7f} (mse {loss_dict_ppg['mse'].item():>7f} fft {loss_dict_ppg['fft_loss'].item():>7f} KLD {loss_dict_ppg['KLD'].item():>7f}) | "
                f"ecg_loss: {loss_dict_ecg['loss'].item():>7f} (mse {loss_dict_ecg['mse'].item():>7f}  KLD {loss_dict_ecg['KLD'].item():>7f}) | "
                f"ppg_loss: {loss_dict_ppg['loss'].item():>7f} (mse {loss_dict_ppg['mse'].item():>7f}  KLD {loss_dict_ppg['KLD'].item():>7f}) | "
                f"align(L2+KL): {(latent_align_loss.item() if isinstance(latent_align_loss, torch.Tensor) else latent_align_loss):>7f} | "
                f"cross: {(cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss):>7f} | "
                f"infonce: {(infonce_loss.item() if isinstance(infonce_loss, torch.Tensor) else infonce_loss):>7f}"
            )

        if save_weights_path and iteration % save_interval == 0:
            model_states = {
                "encoder_ecg": encoder_ecg.state_dict(),
                "encoder_ppg": (encoder_ecg.state_dict() if encoder_ppg is None else encoder_ppg.state_dict()),
                "decoder_ecg": decoder_ecg.state_dict(),
                "decoder_ppg": decoder_ppg.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "iteration": iteration,
                "hparams": {
                    "kld_weight": kld_weight,
                    "lambda_align": lambda_align,
                    "lambda_cross": lambda_cross,
                    "lambda_infonce": lambda_infonce,
                },
            }
            save_path = os.path.join(save_weights_path, f"VAE-iter-{iteration}.pth")
            torch.save(model_states, save_path)
            logger.info(f"Saved checkpoint at iteration {iteration}")


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(args.seed, args.cudnn_deterministic)

    args.save_dir = os.path.join(args.save_dir, "mcmed")
    save_weights_path = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(save_weights_path, exist_ok=True)

    logger = logging.getLogger("vae")
    logger.setLevel("INFO")
    fh = logging.FileHandler(
        os.path.join(args.save_dir, "train.log"),
        encoding="utf-8",
    )
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    decoder_cfg = model_cfg.get("decoder", {})

    # Hyperparameters with sensible defaults
    H_ = {
        "lr": float(train_cfg.get("lr", 2e-5)),
        "batch_size": int(train_cfg.get("batch_size", 4)),
        "total_iterations": int(train_cfg.get("total_iterations", 40000)),
        "save_interval": int(train_cfg.get("save_interval", 10000)),
        "log_interval": int(train_cfg.get("log_interval", 20)),
        "kld_weight": float(train_cfg.get("kld_weight", 1e-4)),
        "share_encoder": bool(train_cfg.get("share_encoder", True)),
        "lambda_align": float(train_cfg.get("lambda_align", 1e-2)),
        "lambda_cross": float(train_cfg.get("lambda_cross", 5e-4)),
        "lambda_infonce": float(train_cfg.get("lambda_infonce", 1e-3)),
        "use_fft_loss": bool(train_cfg.get("use_fft_loss", False)),
        "fft_weight": float(train_cfg.get("fft_weight", 1.0)),
        "num_workers": int(train_cfg.get("num_workers", 32)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    logger.info({"train": H_, "data": data_cfg, "decoder": decoder_cfg})
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Build train dataloader
    # train_dataset = ECGDataset("")
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=H_["batch_size"],
    #     shuffle=True,
    #     num_workers=32,
    #     pin_memory=True,
    # )

    train_dir = data_cfg.get("train_dir", "")
    train_dataset = ECGPPGLMDBDataset(
        dataset_dir=train_dir,
        split='train',
        dataset=data_cfg.get('dataset', 'MCMED')
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=H_["batch_size"],
        shuffle=True,
        num_workers=H_["num_workers"],
        pin_memory=H_["pin_memory"],
        collate_fn=train_dataset.collate_fn,
    )

    encoder_ecg = VAE_Encoder().to(device)
    encoder_ppg = None if H_["share_encoder"] else VAE_Encoder().to(device)
    decoder_ecg = VAE_Decoder(**decoder_cfg).to(device)
    decoder_ppg = VAE_Decoder(**decoder_cfg).to(device)

    loss_fn = loss_function
    parameters = list(encoder_ecg.parameters())
    if encoder_ppg is not None:
        parameters += list(encoder_ppg.parameters())
    parameters += list(decoder_ecg.parameters()) + list(decoder_ppg.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=H_["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=H_["total_iterations"],
        eta_min=1e-9
    )

    train_loop(
        train_dataloader,
        encoder_ecg,
        encoder_ppg,
        decoder_ecg,
        decoder_ppg,
        loss_fn,
        optimizer,
        scheduler,
        device,
        kld_weight=H_["kld_weight"],
        lambda_align=H_["lambda_align"],
        lambda_cross=H_["lambda_cross"],
        lambda_infonce=H_["lambda_infonce"],
        use_fft_loss=H_["use_fft_loss"],
        fft_weight=H_["fft_weight"],
        save_weights_path=save_weights_path,
        logger=logger,
        total_iterations=H_["total_iterations"],
        save_interval=H_["save_interval"],
        log_interval=H_["log_interval"],
    )
    logger.info("Training completed!")
