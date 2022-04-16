import torch


def drop_path(x, drop_prob=0.0, training=False):
    """Drop an entire sample feature from batch, which results in "stochastic depth" when used with residual connection.

    Args:
        x (Tensor): input tensor
        drop_prob (float, optional): Probability to drop samples. Defaults to 0.0.
        training (bool, optional): only drop samples during training. Defaults to False.

    Returns:
        Tensor: dropped sample is filled with zero, others rescaled to keep batch mean invariant.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
