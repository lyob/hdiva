import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_analysis import calculate_convnet_receptive_field


def test_rf_calc():
    print("Testing Standard Mode (Alternating Conv-Pool)...")
    # Case 1: 2 layers, k=3, s=1 (conv), k=2, s=2 (pool)
    # Layer 1 Conv: RF = 1 + (3-1)*1 = 3, stride = 1
    # Layer 1 Pool: RF = 3 + (2-1)*1 = 4, stride = 2
    # Layer 2 Conv: RF = 4 + (3-1)*2 = 8, stride = 2
    # Layer 2 Pool: RF = 8 + (2-1)*2 = 10, stride = 4
    rf = calculate_convnet_receptive_field(
        num_layers=2,
        conv_kernel_size=3,
        conv_stride=1,
        pool_kernel_size=2,
        pool_stride=2,
        pooling_at_end=False,
        verbose=True,
    )
    print(f"Result: {rf}")
    assert rf == 10, f"Expected 10, got {rf}"

    print("\nTesting Pooling At End Mode...")
    # Case 2: 2 layers, k=3, s=1 (conv), k=2, s=2 (pool)
    # Layer 1 Conv: RF = 1 + (3-1)*1 = 3, stride = 1
    # Layer 2 Conv: RF = 3 + (3-1)*1 = 5, stride = 1
    # Final Pool: RF = 5 + (2-1)*1 = 6, stride = 2
    rf = calculate_convnet_receptive_field(
        num_layers=2,
        conv_kernel_size=3,
        conv_stride=1,
        pool_kernel_size=2,
        pool_stride=2,
        pooling_at_end=True,
        verbose=True,
    )
    print(f"Result: {rf}")
    assert rf == 6, f"Expected 6, got {rf}"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_rf_calc()
