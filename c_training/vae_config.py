from dataclasses import dataclass, field

@dataclass
class VAE_Training_Config:
    '''general model parameters'''
    model_name: str = "vae"
    input_dim: int = 32
    in_channels: int = 1
    z_dim: int = 10

    '''dataset parameters'''
    dataset_name: str = "hdisks3"
    dataset_size: int = 5e4
    
    '''Encoder/Decoder parameters'''
    num_blocks: int = 2
    channels: list[int] = field(default_factory=lambda: [1, 32, 32])
    
    '''training'''
    num_epochs: int = 2000
    train_batch_size_per_gpu: int = 512
    learning_rate_init: float = 2e-3
    learning_rate_final: float = 2e-3
    num_warmup_steps: int = 1000
    seed: int = 43

    '''kl schedule'''
    kl_annealing_schedule: str = "constant"
    kl_annealing_epochs: int = 1000
    kl_weight_min: float = 1
    kl_weight_max: float = 1

    '''cluster'''
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_gpus_per_node: int = 4
    num_nodes: int = 1
    precision: str = "32"

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10