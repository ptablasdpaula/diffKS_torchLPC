import json
import torch
import os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def midi_to_hz(midi : torch.Tensor) -> torch.Tensor:
    return 440 * 2 ** ((midi - 69) / 12)

def load_config():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return (
        config["hyperparameters"],
        config["model_params"],
        config["in_domain_params"],
        config["global_settings"],
    )

def resize_tensor_dim(tensor, size, dim=0, pad_value=0):
    """Resize tensor along dimension to target size by padding or cutting."""
    curr_size = tensor.size(dim)

    if curr_size == size:
        return tensor

    # Get the shape of the output tensor
    new_shape = list(tensor.shape)
    new_shape[dim] = size

    # Create a new tensor of the desired shape, filled with pad_value
    result = torch.full(new_shape, pad_value, dtype=tensor.dtype, device=tensor.device)

    if curr_size > size:
        # Copy what fits from the original tensor
        idx_src = [slice(None)] * tensor.ndim
        idx_src[dim] = slice(0, size)
        result = tensor[tuple(idx_src)]
    else:
        # Copy the entire original tensor into the new one
        idx_dst = [slice(None)] * tensor.ndim
        idx_dst[dim] = slice(0, curr_size)
        result[tuple(idx_dst)] = tensor

    return result


def noise_burst(
        sample_rate: int,
        length_s: float,
        burst_width_s: float,
        normalize: bool = False,
        batch_size: int = 1
) -> torch.Tensor: # [batch_size, samples]
    """
    Generate a single-channel noise burst and zero-pad it.
    """
    burst_width_n = int(sample_rate * burst_width_s)
    total_length_n = int(sample_rate * length_s)

    if total_length_n < burst_width_n:
        raise ValueError(
            f"Requested total length {length_s:.3f}s < noise burst width {burst_width_s:.3f}s."
        )

    burst = torch.rand((batch_size, burst_width_n), device=get_device()) - 0.5

    if normalize:
        burst = burst - burst.mean()
        burst = burst / burst.abs().max()

    return resize_tensor_dim(burst, total_length_n, 1)