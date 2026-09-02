import torch


def calculate_simple_convnet_receptive_field(
    num_layers: int,
    conv_kernel_size: int = 3,
    conv_stride: int = 2,
    verbose: bool = False,
) -> int:
    """
    Calculates the receptive field of a Simple ConvNet with only Conv layers.

    Args:
        num_layers: Number of conv layers.
        conv_kernel_size: Kernel size of the convolution layer.
        conv_stride: Stride of the convolution layer.
        verbose: If True, prints the RF and stride after each layer.

    Returns:
        The total receptive field.
    """
    rf = 1
    stride = 1

    if verbose:
        print(f"Initial: RF={rf}, Stride={stride}")

    for i in range(num_layers):
        # Conv layer
        rf = rf + (conv_kernel_size - 1) * stride
        stride = stride * conv_stride
        if verbose:
            print(f"After Layer {i + 1} Conv: RF={rf}, Stride={stride}")

    return rf


def calculate_convnet_receptive_field(
    num_layers: int,
    conv_kernel_size: int = 3,
    conv_stride: int = 2,
    pool_kernel_size: int = 2,
    pool_stride: int = 2,
    pooling_at_end: bool = False,
    verbose: bool = False,
) -> int:
    """
    Calculates the receptive field of a ConvNet with alternating Conv and Pool layers.

    Args:
        num_layers: Number of conv-pool blocks.
        conv_kernel_size: Kernel size of the convolution layer.
        conv_stride: Stride of the convolution layer.
        pool_kernel_size: Kernel size of the pooling layer.
        pool_stride: Stride of the pooling layer.
        pooling_at_end: If True, pooling is applied only once at the end.
        verbose: If True, prints the RF and stride after each layer.

    Returns:
        The total receptive field.
    """
    rf = 1
    stride = 1

    if verbose:
        print(f"Initial: RF={rf}, Stride={stride}")

    for i in range(num_layers):
        # Conv layer
        rf = rf + (conv_kernel_size - 1) * stride
        stride = stride * conv_stride
        if verbose:
            print(f"After Layer {i + 1} Conv: RF={rf}, Stride={stride}")

        # Pool layer
        if not pooling_at_end:
            rf = rf + (pool_kernel_size - 1) * stride
            stride = stride * pool_stride
            if verbose:
                print(f"After Layer {i + 1} Pool: RF={rf}, Stride={stride}")

    if pooling_at_end:
        rf = rf + (pool_kernel_size - 1) * stride
        stride = stride * pool_stride
        if verbose:
            print(f"After Final Pool: RF={rf}, Stride={stride}")

    return rf


def count_parameters(model: torch.nn.Module, verbose: bool = False) -> int:
    """
    Counts the number of parameters in a model.

    Args:
        model: The model to count parameters for.
        verbose: If True, prints the number of parameters.

    Returns:
        The total number of parameters.
    """
    count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Number of parameters: {count}")
    return count
