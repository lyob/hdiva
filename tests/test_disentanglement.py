import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.disentanglement_metrics import measure_disentanglement


# Dummy Encoder
class DummyEncoder(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # Dummy parameter to have a device
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, x):
        # Return random representations or identity if x matches z_dim
        # Here we simulate a perfect disentanglement scenario where z = factors
        # x is expected to be the factors themselves for this dummy test
        return x, torch.zeros_like(x)


# Create dummy dataset
# Factors: 3 dimensions, 1000 samples
num_samples = 1000
z_dim = 3

# Use continuous factors (floats) to test the fix
factors = torch.randint(0, 5, (num_samples, z_dim)).float()
images = factors

dataset = TensorDataset(images, factors)
dataloader = DataLoader(dataset, batch_size=64)

encoder = DummyEncoder(z_dim)

# Capture warnings to see if the specific warning is raised
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    print("Testing Modularity with continuous-looking factors...")
    try:
        mod_scores = measure_disentanglement(encoder, dataloader, "Modularity", num_train=500, num_test=500)
        print("Modularity Scores:", mod_scores)

        # Check for the specific warning
        found_warning = False
        for warning in w:
            if "'multi_class' was deprecated" in str(warning.message):
                found_warning = True
                print("WARNING FOUND: 'multi_class' was deprecated...")
                break

        if not found_warning:
            print("SUCCESS: No 'multi_class' deprecation warning found.")

        # Expect high modularity
        if mod_scores["modularity_score"] > 0.8:
            print("Modularity Test Passed!")
        else:
            print("Modularity Test Failed: Score too low for perfect encoding.")

    except Exception as e:
        print(f"Modularity Test Failed with error: {e}")
        import traceback

        traceback.print_exc()
