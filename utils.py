import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio

def get_device():
    r"""Output 'cuda' if gpu is available, 'cpu' otherwise"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path="config.json"):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return (
        config["hyperparameters"],
        config["model_params"],
        config["in_domain_params"],
        config["global_settings"],
    )

def compute_minimum_action(raw_coeff_frames: torch.Tensor, distance: str = "l2") -> torch.Tensor:
    """
    Penalize large changes in reflection coefficients across adjacent frames.
    raw_coeff_frames is shape [n_frames, 2].
    """
    # shape = [n_frames - 1, 2]
    diffs = raw_coeff_frames[1:] - raw_coeff_frames[:-1]

    if distance.lower() == "l1":
        return torch.abs(diffs).mean()
    else:
        # default to L2
        return (diffs ** 2).mean()

def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform"):
    """
    Plot a waveform (possibly multi-channel; in that case, only plot the first channel).
    Expects waveform shape to be at least 1D, e.g. [time], [channels, time],
    or [batch, channels, time].
    """
    # Squeeze batch dimension if present
    if waveform.dim() == 3:
        # shape: [batch, channels, time]
        waveform = waveform[0]  # take the first item in the batch
    if waveform.dim() == 2:
        # shape: [channels, time]
        waveform = waveform[0]  # take the first channel

    n_samples = waveform.shape[-1]
    time_axis = np.linspace(0, n_samples / sample_rate, n_samples)

    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, waveform.numpy(), label=title, alpha=0.75)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalize waveform to [-1, 1] along the time dimension,
    assuming input shape is [batch, channels, time] or similar.
    Uses infinity norm to scale the largest absolute value to 1.
    """
    # If the audio is [time], turn it into [1, 1, time] for consistent normalization
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        # If it's [channels, time], add a batch dimension
        waveform = waveform.unsqueeze(0)

    # Normalize in-place along the time axis (dim=2 if shape = [batch, channels, time])
    normalized = F.normalize(waveform, p=float("inf"), dim=2)
    return normalized


def save_audio_torchaudio(
    waveform: torch.Tensor,
    sample_rate: int,
    out_path: str,
):
    """
    Save the waveform with torchaudio, ensuring shape is [channels, time].
    """
    # Expecting shape [batch, channels, time] or [channels, time]
    if waveform.dim() == 3:
        # shape: [batch, channels, time]
        waveform = waveform[0]  # Take the first item in the batch

    # Now shape is [channels, time]
    if waveform.dim() == 1:
        # If it was [time], unsqueeze to have [1, time]
        waveform = waveform.unsqueeze(0)

    torchaudio.save(out_path, waveform, sample_rate)


def make_symmetric_mirrored_coefficient_frame_linspace(
    n_frames: int,
    b_start: float,
    b_mid: float,
    b_end: float
) -> torch.Tensor:
    """
    Construct mirrored reflection coefficients in a three segment linspace.
    """
    one_third_frames = n_frames // 3
    two_third_frames = one_third_frames * 2

    # b1 ramps from b_start -> b_mid -> b_end
    b1_first_segment = torch.linspace(b_start, b_mid, two_third_frames)
    b1_second_segment = torch.linspace(b_mid, b_end, n_frames - two_third_frames)
    b1_coeff_frames = torch.cat([b1_first_segment, b1_second_segment])

    # b2 ramps from b_end -> b_mid -> b_start
    b2_first_segment = torch.linspace(b_end, b_mid, one_third_frames)
    b2_second_segment = torch.linspace(b_mid, b_start, n_frames - one_third_frames)
    b2_coeff_frames = torch.cat([b2_first_segment, b2_second_segment])

    return torch.stack([b1_coeff_frames, b2_coeff_frames], dim=1)


def ks_to_audio(
    model: torch.nn.Module,
    out_path: str,
    f0_frames: torch.Tensor,
    sample_rate: int,
    length_audio_s: int,
) -> torch.Tensor:
    """
    Run the model to produce audio, normalize it, plot it, and then
    save it as a WAV file using torchaudio.
    """
    n_samples = sample_rate * length_audio_s

    with torch.no_grad():
        # Generate audio
        time_signal = model(
            delay_len_frames=f0_frames,
            n_samples=n_samples
        ).cpu()

        print(f"Model output shape: {time_signal.shape}")

        # Normalize audio to [-1, 1]
        normalized_signal = normalize_audio(time_signal)

        # Plot the normalized waveform
        plot_waveform(normalized_signal, sample_rate, title=out_path)

        # Save using torchaudio
        save_audio_torchaudio(normalized_signal, sample_rate, out_path)

    return normalized_signal


def process_target(
    target_path : str,
    out_path: str,
    target_sample_rate : int =16000,
    target_length_samples : int=16000 * 4
):
    """
    Process an external audio file to be used as a target for optimization:
      - Load and resample (if needed)
      - Zero-pad or truncate to match desired length
      - Normalize
      - Plot the result
      - Save as a reference (optional)
    """
    # Load the audio file
    waveform, file_sr = torchaudio.load(target_path)

    # Resample if necessary
    if file_sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Ensure the audio is the right length if specified
    if target_length_samples is not None:
        if waveform.shape[1] > target_length_samples:
            waveform = waveform[:, :target_length_samples]
        elif waveform.shape[1] < target_length_samples:
            padding = target_length_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

    # Add a batch dimension => now [1, channels, time]
    waveform = waveform.unsqueeze(0)

    # Normalize to [-1, 1]
    processed_target = F.normalize(waveform, p=float('inf'), dim=2)

    print(f"Processed target shape: {processed_target.shape}")

    # Plot the processed audio
    plot_waveform(processed_target, target_sample_rate, title=out_path)

    # (Optional) Save the processed target for reference
    torchaudio.save(out_path, processed_target[0], target_sample_rate)

    return processed_target
