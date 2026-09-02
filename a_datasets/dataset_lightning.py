import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms

from a_datasets.bump.data import load_bump_data
from a_datasets.circle_in_square.data import load_cis_data
from a_datasets.custom_dataset_classes import CustomTensorDataset
from a_datasets.dsprites.dsprites import load_dsprites_data
from a_datasets.sos.data import load_sos_data
from a_datasets.disks.data import load_simple_disk_dataset


class GeneralDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.transform = transforms.Compose(
            [
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self):
        """
        Download data. This is called once on the main process.
        """
        # This will download and cache the dataset. It won't be called by other GPUs.
        if self.config.dataset_name == "bump":
            load_bump_data(holdout_center=self.config.holdout_center)
        elif self.config.dataset_name == "sos":
            load_sos_data(holdout_center=self.config.holdout_center)
        elif self.config.dataset_name == "simple_disks":
            load_simple_disk_dataset(
                num_samples=self.config.dataset_size,
                img_size=self.config.image_dim, 
                foreground=self.config.foreground,
                background=self.config.background,
                outer_radius=self.config.outer_radius, 
                transition_width=self.config.transition_width,
                seed=self.config.dataset_seed,
                cx_range=self.config.cx_range,
                cy_range=self.config.cy_range,
            )
        elif self.config.dataset_name == "dsprites":
            load_dsprites_data()
        elif self.config.dataset_name == "circle_in_square":
            load_cis_data(img_size=self.config.image_dim)
        else:
            raise ValueError(f"Unknown dataset name: {self.config.dataset_name}")

    def setup(self, stage: str):
        """
        Download, prepare, and split data. This is called once per GPU.
        The data remains on the CPU.
        """
        if stage == "fit":
            if self.config.dataset_name == "bump":
                cpu_tensor, _ = load_bump_data(holdout_center=self.config.holdout_center)
            elif self.config.dataset_name == "sos":
                cpu_tensor, _ = load_sos_data(
                    holdout_center=self.config.holdout_center, sigma_x=self.config.sigma_x, sigma_y=self.config.sigma_y
                )
            elif self.config.dataset_name == "simple_disks":
                cpu_tensor, _ = load_simple_disk_dataset(
                    num_samples=self.config.dataset_size,
                    img_size=self.config.image_dim, 
                    foreground=self.config.foreground,
                    background=self.config.background,
                    outer_radius=self.config.outer_radius, 
                    transition_width=self.config.transition_width,
                    seed=self.config.dataset_seed,
                )
            elif self.config.dataset_name == "dsprites":
                cpu_tensor = load_dsprites_data()
            elif self.config.dataset_name == "circle_in_square":
                cpu_tensor, _ = load_cis_data()
            else:
                raise ValueError(f"Unknown dataset name: {self.config.dataset_name}")
            self.train_dataset = CustomTensorDataset(cpu_tensor, self.transform)
            print(f"Dataset setup complete.")

    def train_dataloader(self):
        """
        This creates the DataLoader that will fetch batches from the CPU dataset.
        """
        # num_workers > 0 spins up subprocesses to load data in the background from the CPU.
        # This prevents the GPU from waiting for data. A good starting point is os.cpu_count().
        num_workers = 4 * self.config.num_gpus_per_node  # Your original logic is good

        # pin_memory=True speeds up the CPU-to-GPU memory transfer.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size_per_gpu,
            shuffle=True,
            num_workers=num_workers,  # <-- Use multiple workers
            pin_memory=True,  # <-- Set to True for GPU training
            persistent_workers=True if num_workers > 0 else False,
        )
