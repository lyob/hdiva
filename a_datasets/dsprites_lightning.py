from torch.utils.data import DataLoader
import lightning as L
from a_datasets.custom_dataset_classes import CustomTensorDataset
from a_datasets.dsprites import load_dsprites_data

from torchvision import transforms

class DspritesDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def prepare_data(self):
        """
        Download data. This is called once on the main process.
        """
        # This will download and cache the dataset. It won't be called by other GPUs.
        load_dsprites_data()

    def setup(self, stage: str):
        """
        Download, prepare, and split data. This is called once per GPU.
        The data remains on the CPU.
        """
        if stage == "fit":
            # cpu_tensor = load_dataset("flwrlabs/celeba", split='train')
            cpu_tensor = load_dsprites_data()
            self.train_dataset = CustomTensorDataset(cpu_tensor, self.transform)
            print(f"Dataset setup complete.")

    def train_dataloader(self):
        """
        This creates the DataLoader that will fetch batches from the CPU dataset.
        """
        # num_workers > 0 spins up subprocesses to load data in the background from the CPU.
        # This prevents the GPU from waiting for data. A good starting point is os.cpu_count().
        num_workers = 4 * self.config.num_gpus_per_node # Your original logic is good

        # pin_memory=True speeds up the CPU-to-GPU memory transfer.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size_per_gpu,
            shuffle=True,
            num_workers=num_workers,    # <-- Use multiple workers
            pin_memory=True,            # <-- Set to True for GPU training
            persistent_workers=True if num_workers > 0 else False
        )