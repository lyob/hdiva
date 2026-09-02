from dataclasses import dataclass, field

@dataclass
class LVAE_Training_Config:
    '''general model parameters'''
    model_name: str = "lvae"
    input_dim: int = 32
    in_channels: int = 1
    z_dims: list[int] = field(default_factory=lambda: [6, 4])

    '''dataset parameters'''
    dataset_name: str = "hdisks3"
    dataset_size: int = 5e4
    
    '''Encoder/Decoder parameters'''
    num_blocks: int = 2
    enc_channels: list[int] = field(default_factory=lambda: [1, 32, 64])
    dec_channels: list[int] = field(default_factory=lambda: [64, 32, 1])
    dec_skip_connections: bool = True

    '''training'''
    num_epochs: int = 4000
    train_batch_size_per_gpu: int = 512
    lr_schedule = "linear"
    lr_init: float = 3e-4
    lr_final: float = 2e-5
    seed: int = 43
    use_nll_loss: bool = True

    '''kl schedule'''
    kl_annealing_schedule: str = "linear"
    kl_annealing_epochs: int = 800
    kl_weights_min: list[float] = field(default_factory=lambda: [0, 0])
    kl_weights_max: list[float] = field(default_factory=lambda: [1, 1])

    '''cluster'''
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_gpus_per_node: int = 4
    num_nodes: int = 1
    precision: str = "32"

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10