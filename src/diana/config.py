import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_SCHEDULES = {"linear", "cosine"}
VALID_OBJECTIVES = {"pred_noise", "pred_x0", "pred_v"}
VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}


def load_toml_config(path: str) -> dict[str, Any]:
    """Parse a flat TOML config file whose top-level keys mirror Config fields.

    Unknown keys are rejected up front so typos surface as a clear error
    instead of a confusing TypeError deep in Config construction.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ValueError(f"config file not found: {path}")
    with cfg_path.open("rb") as f:
        data = dict(tomllib.load(f))
    valid = {field.name for field in dataclasses.fields(Config)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"unknown key(s) in config file {path!r}: {', '.join(unknown)}; "
            f"valid keys: {', '.join(sorted(valid))}"
        )
    return data


@dataclass
class Config:
    # Data
    data_path: str = ""
    category: str = "default"
    img_size: int = 64
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    augment_hflip: bool = False

    # Model
    base_channels: int = 64
    channel_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [16])
    dropout: float = 0.1
    ema_decay: float = 0.9999

    # Diffusion
    schedule_type: str = "linear"
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_param: float = 0.0
    objective: str = "pred_noise"
    sample_timesteps: int = 250

    # Optim
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # HPC
    device: str = "auto"
    seed: int = 42
    use_amp: bool = False
    cudnn_benchmark: bool = False
    use_cuda_graphs: bool = False
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 25
    resume_from: str | None = None
    log_every_n_steps: int = 50

    def __post_init__(self):
        # Data
        if not self.data_path:
            raise ValueError("data_path must not be empty")
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

        # Model
        if self.base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {self.base_channels}"
            )
        if not self.channel_mults or any(m <= 0 for m in self.channel_mults):
            raise ValueError(
                f"channel_mults must be a non-empty list of positive ints, got {self.channel_mults}"
            )
        if self.channel_mults[0] != 1:
            raise ValueError(
                f"channel_mults must start with 1, got {self.channel_mults}"
            )
        if self.num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {self.num_res_blocks}")
        n_levels = len(self.channel_mults)
        if self.img_size % (2**n_levels) != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by 2**len(channel_mults) "
                f"({2**n_levels}) so the network's smallest resolution is an integer"
            )
        visited_resolutions = {self.img_size // 2**i for i in range(n_levels)}
        if not self.attention_resolutions or any(
            a <= 0 for a in self.attention_resolutions
        ):
            raise ValueError(
                f"attention_resolutions must be a non-empty list of positive ints, got {self.attention_resolutions}"
            )
        if any(a not in visited_resolutions for a in self.attention_resolutions):
            raise ValueError(
                f"attention_resolutions {self.attention_resolutions} must be a subset of "
                f"the resolutions the network visits {sorted(visited_resolutions)}"
            )
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")

        # Diffusion
        if self.schedule_type not in VALID_SCHEDULES:
            raise ValueError(
                f"schedule_type must be one of {sorted(VALID_SCHEDULES)}, got {self.schedule_type!r}"
            )
        if self.num_timesteps <= 0:
            raise ValueError(
                f"num_timesteps must be positive, got {self.num_timesteps}"
            )
        if self.beta_start <= 0:
            raise ValueError(f"beta_start must be positive, got {self.beta_start}")
        if self.beta_end <= self.beta_start:
            raise ValueError(
                f"beta_end must be > beta_start ({self.beta_start}), got {self.beta_end}"
            )
        if self.beta_end >= 1.0:
            raise ValueError(f"beta_end must be < 1.0, got {self.beta_end}")
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(VALID_OBJECTIVES)}, got {self.objective!r}"
            )
        if self.sample_timesteps <= 0:
            raise ValueError(
                f"sample_timesteps must be positive, got {self.sample_timesteps}"
            )
        if self.sample_timesteps > self.num_timesteps:
            raise ValueError(
                f"sample_timesteps ({self.sample_timesteps}) must not exceed num_timesteps ({self.num_timesteps})"
            )
        if self.num_timesteps % self.sample_timesteps != 0:
            raise ValueError(
                f"num_timesteps ({self.num_timesteps}) must be divisible by sample_timesteps "
                f"({self.sample_timesteps}) so the reverse chain can be strided evenly"
            )

        # Optim
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.lr_warmup_steps < 0:
            raise ValueError(
                f"lr_warmup_steps must be >= 0, got {self.lr_warmup_steps}"
            )
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )

        # HPC
        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(VALID_DEVICES)}, got {self.device!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.save_every_n_epochs < 1:
            raise ValueError(
                f"save_every_n_epochs must be >= 1, got {self.save_every_n_epochs}"
            )
        if self.log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be >= 1, got {self.log_every_n_steps}"
            )


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Configuration for Diffusion Model Training"
    )

    # ==========================
    # General Arguments
    # ==========================
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file; explicitly-set CLI flags override its values",
    )

    # ==========================
    # Data Arguments
    # ==========================
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Dataset root containing <category>/train/good (or 'synthetic')",
    )
    data_group.add_argument(
        "--category", type=str, default=argparse.SUPPRESS, help="Dataset category"
    )
    data_group.add_argument(
        "--img_size", type=int, default=argparse.SUPPRESS, help="Image resolution"
    )
    data_group.add_argument(
        "--batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size per GPU",
    )
    data_group.add_argument(
        "--num_workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of dataloader workers",
    )
    data_group.add_argument(
        "--pin_memory",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Pin memory for dataloaders",
    )
    data_group.add_argument(
        "--augment_hflip",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable horizontal flip augmentation",
    )

    # ==========================
    # Model Arguments
    # ==========================
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--base_channels",
        type=int,
        default=argparse.SUPPRESS,
        help="Base channel count for UNet",
    )
    # Using nargs='+' allows passing multiple values like: --channel_mults 1 2 4 8
    model_group.add_argument(
        "--channel_mults",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Channel multipliers",
    )
    model_group.add_argument(
        "--num_res_blocks",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of residual blocks per level",
    )
    model_group.add_argument(
        "--attention_resolutions",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Resolutions to apply attention",
    )
    model_group.add_argument(
        "--dropout", type=float, default=argparse.SUPPRESS, help="Dropout probability"
    )
    model_group.add_argument(
        "--ema_decay",
        type=float,
        default=argparse.SUPPRESS,
        help="Exponential Moving Average decay",
    )

    # ==========================
    # Diffusion Arguments
    # ==========================
    diffusion_group = parser.add_argument_group("Diffusion")
    diffusion_group.add_argument(
        "--schedule_type",
        type=str,
        default=argparse.SUPPRESS,
        choices=["linear", "cosine"],
        help="Noise schedule type",
    )
    diffusion_group.add_argument(
        "--num_timesteps",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of diffusion timesteps",
    )
    diffusion_group.add_argument(
        "--beta_start",
        type=float,
        default=argparse.SUPPRESS,
        help="Starting beta value",
    )
    diffusion_group.add_argument(
        "--beta_end", type=float, default=argparse.SUPPRESS, help="Ending beta value"
    )
    diffusion_group.add_argument(
        "--schedule_param",
        type=float,
        default=argparse.SUPPRESS,
        help="Extra parameter for schedule (if needed)",
    )
    diffusion_group.add_argument(
        "--objective",
        type=str,
        default=argparse.SUPPRESS,
        choices=["pred_noise", "pred_x0", "pred_v"],
        help="Diffusion objective",
    )
    diffusion_group.add_argument(
        "--sample_timesteps",
        type=int,
        default=argparse.SUPPRESS,
        help="Timesteps to use during sampling/inference",
    )

    # ==========================
    # Optim Arguments
    # ==========================
    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--epochs", type=int, default=argparse.SUPPRESS, help="Total training epochs"
    )
    optim_group.add_argument(
        "--learning_rate", type=float, default=argparse.SUPPRESS, help="Learning rate"
    )
    optim_group.add_argument(
        "--weight_decay", type=float, default=argparse.SUPPRESS, help="Weight decay"
    )
    optim_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of warmup steps for LR scheduler",
    )
    optim_group.add_argument(
        "--grad_clip",
        type=float,
        default=argparse.SUPPRESS,
        help="Gradient clipping threshold",
    )
    optim_group.add_argument(
        "--grad_accum_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Gradient accumulation steps",
    )

    # ==========================
    # HPC Arguments
    # ==========================
    hpc_group = parser.add_argument_group("HPC & Hardware")
    hpc_group.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Device to run on (cuda/cpu)",
    )
    hpc_group.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Random seed for reproducibility",
    )
    hpc_group.add_argument(
        "--use_amp",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Automatic Mixed Precision (AMP)",
    )
    hpc_group.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable cuDNN benchmark",
    )
    hpc_group.add_argument(
        "--use_cuda_graphs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable CUDA graphs for faster execution",
    )
    hpc_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Root directory for run outputs and checkpoints",
    )
    hpc_group.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=argparse.SUPPRESS,
        help="Checkpoint cadence: save a rotating checkpoint every N epochs",
    )
    hpc_group.add_argument(
        "--resume_from",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to checkpoint to resume training from",
    )
    hpc_group.add_argument(
        "--log_every_n_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Logging interval in steps",
    )

    ns = vars(parser.parse_args(argv))
    config_path = ns.pop("config", None)

    # Precedence: CLI > TOML > defaults. argparse.SUPPRESS means unset flags
    # never appear in ns, so only explicitly-given flags land in overrides.
    merged: dict[str, Any] = {}
    if config_path:
        merged.update(load_toml_config(config_path))
    merged.update({k: v for k, v in ns.items() if v is not None})
    return Config(**merged)
