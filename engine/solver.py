import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from model.cardioalign_encoder.cardioalign_model import VAE_Decoder, VAE_Encoder
from utils.io_utils import instantiate_from_config, get_model_parameters_info

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

def cycle(dl):
    while True:
        for data in dl:
            yield data

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(move_to_device(d, device) for d in data)
    elif isinstance(data, list):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.device
        # --- solver hyperparameters from YAML ---
        solver_cfg = config.get("solver", {})
        ema_cfg = solver_cfg.get("ema", {})
        opt_cfg = solver_cfg.get("optimizer", {})
        vae_cfg = solver_cfg.get("vae", {})

        self.train_num_steps = int(solver_cfg.get("max_steps", 10000))
        self.gradient_accumulate_every = int(solver_cfg.get("gradient_accumulate_every", 1))
        self.save_cycle = int(solver_cfg.get("save_cycle", 1000))
        self.dl = cycle(dataloader)
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger
        self.results_folder = Path(
            config["solver"]["results_folder"]) / "checkpoints"
        os.makedirs(self.results_folder, exist_ok=True)
        self.use_text = bool(solver_cfg.get("use_text", False))

        start_lr = float(solver_cfg.get("base_lr", 1.0e-4))
        ema_decay = float(ema_cfg.get("decay", 0.995))
        ema_update_every = int(ema_cfg.get("update_interval", 10))
        self.log_frequency = int(solver_cfg.get("log_frequency", 100))
        self.clip_grad_max_norm = float(solver_cfg.get("clip_grad_norm", 1.0))
        opt_betas = opt_cfg.get("betas", [0.9, 0.96])

        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=opt_betas,
        )
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        # VAE checkpoints (for latent mapping)
        self.vae_checkpoint_path = vae_cfg.get(
            "checkpoint",
            "",
        )
        self.vae_encoder_ecg, self.vae_decoder_ecg = self.load_vae_ecg()
        self.vae_encoder_ppg, self.vae_decoder_ppg = self.load_vae_ppg()

        sc_cfg = config["solver"]["scheduler"]
        sc_cfg["params"]["optimizer"] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))

    def load_vae_ecg(self):
        vae_checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        vae_encoder = VAE_Encoder().to(self.device)
        vae_decoder = VAE_Decoder().to(self.device)
        vae_encoder.load_state_dict(vae_checkpoint["encoder_ecg"])
        vae_decoder.load_state_dict(vae_checkpoint["decoder_ecg"])
        for param in vae_encoder.parameters():
            param.requires_grad = False
        for param in vae_decoder.parameters():
            param.requires_grad = False
        vae_encoder.eval()
        vae_decoder.eval()

        return vae_encoder, vae_decoder

    def load_vae_ppg(self):
        vae_checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        vae_encoder = VAE_Encoder().to(self.device)
        vae_decoder = VAE_Decoder().to(self.device)
        vae_encoder.load_state_dict(vae_checkpoint["encoder_ecg"])
        vae_decoder.load_state_dict(vae_checkpoint["decoder_ppg"])
        for param in vae_encoder.parameters():
            param.requires_grad = False
        for param in vae_decoder.parameters():
            param.requires_grad = False
        vae_encoder.eval()
        vae_decoder.eval()

        return vae_encoder, vae_decoder

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                "Save current model to {}".format(
                    str(self.results_folder / f"checkpoint-{milestone}.pt")
                )
            )
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder /
                   f"checkpoint-{milestone}.pt"))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                "Resume from {}".format(
                    str(self.results_folder / f"checkpoint-{milestone}.pt")
                )
            )
        device = self.device
        data = torch.load(
            str(self.results_folder / f"checkpoint-{milestone}.pt"), map_location=device
        )
        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.milestone = milestone

        self.model.eval()

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(
                "{}: start training...".format(self.args.name), check_primary=False
            )

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    ppg, ecg = data[1], data[2]
                    ppg, ecg = ppg.unsqueeze(-1).to(device).float(), ecg.unsqueeze(-1).to(device).float()
                    report = None

                    with torch.no_grad():
                        # Map input ECG to latent space
                        # print(ecg.shape,ppg.shape)
                        latent_data = self.vae_encoder_ecg(ecg)[0]
                        # print("latent_data:",latent_data.shape)
                        latent_target = latent_data
                        # print("latent_target:",latent_target.shape)
                        latent_cond = self.vae_encoder_ppg(ppg)[0]
                        # print("latent_cond:",latent_cond.shape)

                    loss = self.model(
                        x=latent_data,
                        target=latent_target,
                        cond=latent_cond,
                        report=report,
                    )
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                current_lr = self.opt.param_groups[0]['lr']
                pbar.set_description(f"loss: {total_loss:.6f}, lr: {current_lr:.4e}")

                clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        self.logger.log_info(
                            "saved in {}".format(
                                str(
                                    self.results_folder
                                    / f"checkpoint-{self.milestone}.pt"
                                )
                            )
                        )
                        self.logger.log_info(f"total_loss: {total_loss}")

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(
                            tag="train/loss",
                            scalar_value=total_loss,
                            global_step=self.step,
                        )

                pbar.update(1)

        print("training complete")
        if self.logger is not None:
            self.logger.log_info(
                "Training done, time: {:.2f}".format(time.time() - tic)
            )

    def sample_shift(
        self,
        raw_dataloader,
        shape=None,
        sampling_steps=10,
        subset_save_threshold=50000,
        save_dir="output/",
    ):
        device = self.device
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to restore...")

        os.makedirs(save_dir, exist_ok=True)
        model_kwargs = {}

        overall_samples = np.empty([0, shape[0], shape[1]])
        overall_reals = np.empty([0, shape[0], shape[1]])
        overall_masks = np.empty([0, shape[0], shape[1]])

        subset_samples = np.empty([0, shape[0], shape[1]])
        subset_reals = np.empty([0, shape[0], shape[1]])
        subset_masks = np.empty([0, shape[0], shape[1]])

        file_idx = 0
        current_sample_count = 0

        # for idx, (ppg, ecg) in enumerate(
        for idx, (ppg, ecg, label) in enumerate(
            tqdm(raw_dataloader, total=len(raw_dataloader), desc="Restoring")
        ):
            ppg, ecg = ppg.unsqueeze(-1).to(device).float(), ecg.unsqueeze(-1).to(device).float()
            report = None

            if report is not None:
                report = move_to_device(report, device)

            with torch.no_grad():
                # Map input ECG to latent space
                latent_data = self.vae_encoder_ecg(ecg)[0]
                latent_cond = self.vae_encoder_ppg(ppg)[0]


            sample = self.ema.ema_model.sample_shift(
                shape=latent_data.shape,
                cond=latent_cond,
                report=report,
            )

            with torch.no_grad():
                sample = self.vae_decoder_ecg(sample)

            overall_samples = np.row_stack(
                [overall_samples, sample.detach().cpu().numpy()]
            )
            overall_reals = np.row_stack(
                [overall_reals, ecg.detach().cpu().numpy()])
            overall_masks = np.row_stack(
                [overall_masks, ppg.detach().cpu().numpy()])

            subset_samples = np.row_stack(
                [subset_samples, sample.detach().cpu().numpy()]
            )
            subset_reals = np.row_stack(
                [subset_reals, ecg.detach().cpu().numpy()])
            subset_masks = np.row_stack(
                [subset_masks, ppg.detach().cpu().numpy()])

            current_sample_count += sample.shape[0]

            # Save subset
            if current_sample_count >= subset_save_threshold:
                subset_file = os.path.join(
                    save_dir, f"subset_fake_data_{file_idx}.npy")
                np.save(subset_file, subset_samples)
                if self.logger is not None:
                    self.logger.log_info(
                        f"Saved subset {file_idx} with {current_sample_count} samples to {subset_file}"
                    )

                # reset
                subset_samples = np.empty([0, shape[0], shape[1]])
                subset_reals = np.empty([0, shape[0], shape[1]])
                subset_masks = np.empty([0, shape[0], shape[1]])

                current_sample_count = 0
                file_idx += 1

        if current_sample_count > 0:
            subset_file = os.path.join(
                save_dir, f"subset_fake_data_{file_idx}.npy")
            np.save(subset_file, subset_samples)
            if self.logger is not None:
                self.logger.log_info(
                    f"Saved final subset {file_idx} with {current_sample_count} samples to {subset_file}"
                )

        # Save all
        overall_file = os.path.join(save_dir, "overall_fake_data.npy")
        np.save(overall_file, overall_samples)
        if self.logger is not None:
            self.logger.log_info(
                f"Saved overall data with {overall_samples.shape[0]} samples to {overall_file}"
            )
        overall_file = os.path.join(save_dir, "overall_gt_data.npy")
        np.save(overall_file, overall_reals)
        if self.logger is not None:
            self.logger.log_info(
                "Imputation done, time: {:.2f}".format(time.time() - tic)
            )
        overall_file = os.path.join(save_dir, "overall_gt_ppg_data.npy")
        np.save(overall_file, overall_masks)
        if self.logger is not None:
            self.logger.log_info(
                "Imputation done, time: {:.2f}".format(time.time() - tic)
            )
        return overall_samples, overall_reals, overall_masks

    def restore(
        self,
        raw_dataloader,
        shape=None,
        coef=1e-1,
        learning_rate=1e-1,
        sampling_steps=50,
        subset_save_threshold=50000,
        save_dir="output/",
    ):
        device = self.device
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to restore...")

        os.makedirs(save_dir, exist_ok=True)
        model_kwargs = {}
        model_kwargs["coef"] = coef
        model_kwargs["learning_rate"] = learning_rate

        overall_samples = np.empty([0, shape[0], shape[1]])
        overall_reals = np.empty([0, shape[0], shape[1]])
        overall_masks = np.empty([0, shape[0], shape[1]])

        subset_samples = np.empty([0, shape[0], shape[1]])
        subset_reals = np.empty([0, shape[0], shape[1]])
        subset_masks = np.empty([0, shape[0], shape[1]])

        file_idx = 0
        current_sample_count = 0

        for idx, (x, t_m) in enumerate(
            tqdm(raw_dataloader, total=len(raw_dataloader), desc="Restoring")
        ):
            x = move_to_device(x, device)
            t_m = move_to_device(t_m, device)

            data, report = (x[0], x[1]) if self.use_text else (x, None)
            target = data * t_m
            t_m = t_m.bool()

            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(
                    shape=data.shape,
                    target=target,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    report=report,
                )
            else:
                sample = self.ema.ema_model.fast_sample_infill(
                    shape=data.shape,
                    target=target,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                    report=report,
                )

            overall_samples = np.row_stack(
                [overall_samples, sample.detach().cpu().numpy()]
            )
            overall_reals = np.row_stack(
                [overall_reals, data.detach().cpu().numpy()])
            overall_masks = np.row_stack(
                [overall_masks, t_m.detach().cpu().numpy()])

            subset_samples = np.row_stack(
                [subset_samples, sample.detach().cpu().numpy()]
            )
            subset_reals = np.row_stack(
                [subset_reals, data.detach().cpu().numpy()])
            subset_masks = np.row_stack(
                [subset_masks, t_m.detach().cpu().numpy()])

            current_sample_count += sample.shape[0]

            # Save subset
            if current_sample_count >= subset_save_threshold:
                subset_file = os.path.join(
                    save_dir, f"subset_fake_data_{file_idx}.npy")
                np.save(subset_file, subset_samples)
                if self.logger is not None:
                    self.logger.log_info(
                        f"Saved subset {file_idx} with {current_sample_count} samples to {subset_file}"
                    )

                # reset
                subset_samples = np.empty([0, shape[0], shape[1]])
                subset_reals = np.empty([0, shape[0], shape[1]])
                subset_masks = np.empty([0, shape[0], shape[1]])

                current_sample_count = 0
                file_idx += 1

        if current_sample_count > 0:
            subset_file = os.path.join(
                save_dir, f"subset_fake_data_{file_idx}.npy")
            np.save(subset_file, subset_samples)
            if self.logger is not None:
                self.logger.log_info(
                    f"Saved final subset {file_idx} with {current_sample_count} samples to {subset_file}"
                )

        # Save all
        overall_file = os.path.join(save_dir, "overall_fake_data.npy")
        np.save(overall_file, overall_samples)
        if self.logger is not None:
            self.logger.log_info(
                f"Saved overall data with {overall_samples.shape[0]} samples to {overall_file}"
            )
        if self.logger is not None:
            self.logger.log_info(
                "Imputation done, time: {:.2f}".format(time.time() - tic)
            )

        return overall_samples, overall_reals, overall_masks