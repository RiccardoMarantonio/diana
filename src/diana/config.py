from torch.distributed.launch import parse_args
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_path: str
    category: str
    img_size: str
    batch_size: str
    num_workers: str
    pin_memory: bool
    augment_hflip: str

    # Model	: str
    base_channels: str
    channel_mults: str
    num_res_blocks: str
    attention_resolutions: str
    dropout: str
    ema_decay: str

    # Diffusion	: str
    schedule_type: str
    num_timesteps: str
    beta_start: str
    beta_end: str
    schedule_param: str
    objective: str
    sample_timesteps: str

    # Optim	: str
    epochs: str
    learning_rate: str
    weight_decay: str
    lr_warmup_steps: str
    grad_clip: str
    grad_accum_steps: str

    # HPC	: str
    device: str
    seed: str
    use_amp: str
    cudnn_benchmark: str
    use_cuda_graphs: str
    checkpoint_dir: str
    resume_from: str
    log_every_n_steps: str


def __post_init__(self):
    raise ValueError("CHECKS ARE NOT IMPLEMENTED YET!")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Configuration for Diffusion Model Training"
    )

    # ==========================
    # Data Arguments
    # ==========================
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )
    data_group.add_argument(
        "--category", type=str, default="default", help="Dataset category"
    )
    data_group.add_argument(
        "--img_size", type=int, default=256, help="Image resolution"
    )
    data_group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per GPU"
    )
    data_group.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    data_group.add_argument(
        "--pin_memory", action="store_true", help="Pin memory for dataloaders"
    )
    data_group.add_argument(
        "--augment_hflip",
        action="store_true",
        help="Enable horizontal flip augmentation",
    )

    # ==========================
    # Model Arguments
    # ==========================
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--base_channels", type=int, default=128, help="Base channel count for UNet"
    )
    # Using nargs='+' allows passing multiple values like: --channel_mults 1 2 4 8
    model_group.add_argument(
        "--channel_mults",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Channel multipliers",
    )
    model_group.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="Number of residual blocks per level",
    )
    model_group.add_argument(
        "--attention_resolutions",
        type=int,
        nargs="+",
        default=[16],
        help="Resolutions to apply attention",
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    model_group.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Exponential Moving Average decay",
    )

    # ==========================
    # Diffusion Arguments
    # ==========================
    diffusion_group = parser.add_argument_group("Diffusion")
    diffusion_group.add_argument(
        "--schedule_type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Noise schedule type",
    )
    diffusion_group.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    diffusion_group.add_argument(
        "--beta_start", type=float, default=1e-4, help="Starting beta value"
    )
    diffusion_group.add_argument(
        "--beta_end", type=float, default=0.02, help="Ending beta value"
    )
    diffusion_group.add_argument(
        "--schedule_param",
        type=float,
        default=0.0,
        help="Extra parameter for schedule (if needed)",
    )
    diffusion_group.add_argument(
        "--objective",
        type=str,
        default="pred_noise",
        choices=["pred_noise", "pred_x0", "pred_v"],
        help="Diffusion objective",
    )
    diffusion_group.add_argument(
        "--sample_timesteps",
        type=int,
        default=250,
        help="Timesteps to use during sampling/inference",
    )

    # ==========================
    # Optim Arguments
    # ==========================
    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--epochs", type=int, default=100, help="Total training epochs"
    )
    optim_group.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    optim_group.add_argument(
        "--weight_decay", type=float, default=1e-6, help="Weight decay"
    )
    optim_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for LR scheduler",
    )
    optim_group.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping threshold"
    )
    optim_group.add_argument(
        "--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps"
    )

    # ==========================
    # HPC Arguments
    # ==========================
    hpc_group = parser.add_argument_group("HPC & Hardware")
    hpc_group.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda/cpu)"
    )
    hpc_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    hpc_group.add_argument(
        "--use_amp", action="store_true", help="Use Automatic Mixed Precision (AMP)"
    )
    hpc_group.add_argument(
        "--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark"
    )
    hpc_group.add_argument(
        "--use_cuda_graphs",
        action="store_true",
        help="Enable CUDA graphs for faster execution",
    )
    hpc_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    hpc_group.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    hpc_group.add_argument(
        "--log_every_n_steps", type=int, default=50, help="Logging interval in steps"
    )

    return Config(**vars(parser.parse_args()))
