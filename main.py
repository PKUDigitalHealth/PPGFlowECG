import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)
from utils.ppgecg_dataset import ECGPPGLMDBDataset
from utils.saved_dataset import SavedDataset
from torch.utils.data import Dataset, DataLoader
# huggingface offline mode
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument("--name", type=str, default="latent_rectified_flow")

    parser.add_argument(
        "--config_file", type=str, default="config/latent_rectified_flow.yaml", help="path of config file"
    )
    parser.add_argument(
        "--output", type=str, default="baseline", help="directory to save the results"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard for logging"
    )

    # args for random

    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use. If given, only the specific gpu will be"
        " used, and ddp will be disabled",
    )

    # args for training
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train or Test."
    )
    parser.add_argument(
        "--condition_type",
        type=int,
        default=1,
        choices=[0, 1],
        help="Uncondition or Condition.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthesis",
        help="Infilling, Forecasting or Synthesis.",
    )
    parser.add_argument("--milestone", type=int, default=10)

    parser.add_argument(
        "--synthesis_channels",
        type=lambda x: list(map(int, x.split(","))),
        default=list(range(1, 2)),
        help="List of synthesis channels (default is [1, 2, 3, ..., 11]).",
    )

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f"{args.name}")

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    print(config)

    logger = Logger(args)
    logger.save_config(config)

    # --- read optional data & sampling configs from YAML ---
    data_cfg = config.get("data", {})
    train_data_cfg = data_cfg.get("train", {})
    test_data_cfg = data_cfg.get("test", {})
    sample_cfg = config.get("sample", {})

    if args.train:
        model = instantiate_from_config(config["model"]).cuda()
        train_dir = train_data_cfg.get("dir", "")
        train_dataset = ECGPPGLMDBDataset(
            dataset_dir=train_dir,
            split='train',
            dataset='MCMED'
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=int(train_data_cfg.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(train_data_cfg.get("num_workers", 32)),
            pin_memory=bool(train_data_cfg.get("pin_memory", True)),
            collate_fn=train_dataset.collate_fn
        )
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=train_dataloader,
            logger=logger,
        )
        trainer.train()

    elif args.condition_type == 1 and args.mode in ["synthesis"]:
        model = instantiate_from_config(config["model"]).cuda()
        # Load test dataset (SavedDataset path can be configured via YAML)
        test_saved_dir = test_data_cfg.get('saved_dir', '')
        test_dataset = SavedDataset(data_dir=test_saved_dir, split='test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=int(test_data_cfg.get('batch_size', 128)),
            shuffle=False,
            num_workers=int(test_data_cfg.get('num_workers', 32)),
            pin_memory=bool(test_data_cfg.get('pin_memory', True))
        )
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=test_dataloader,
            logger=logger,
        )

        trainer.load(args.milestone)

        sampling_steps = int(sample_cfg.get('sampling_steps', 10))
        subset_save_threshold = int(sample_cfg.get('subset_save_threshold', 100000))
        samples, *_ = trainer.sample_shift(
            test_dataloader,
            # ours
            [1280,1],
            sampling_steps,
            subset_save_threshold=subset_save_threshold,
            save_dir=os.path.join(args.save_dir, "samples"),
        )

if __name__ == "__main__":
    main()
