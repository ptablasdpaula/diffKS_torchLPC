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

def get_raw_coefficients(self) -> torch.Tensor:
    """
    Returns the unnormalized (pre-activation) output of the coefficient generator.
    Shape: [n_frames, l_filter_order]
    """
    t = torch.linspace(0, 1, steps=self.n_frames, device=self.coeff_generator.weight.device).unsqueeze(1)
    return self.coeff_generator(t)


def compute_minimum_action(coeff_frames: torch.Tensor, distance: str = "l2") -> torch.Tensor:
    """
    Penalize large changes in reflection coefficients across adjacent frames.
    coeff_frames is shape [n_frames, l_filter_order].
    Applies a discrete Laplacian (second difference) smoothing penalty across time.
    """
    diffs = (coeff_frames[1:-1] - coeff_frames[:-2]) - (coeff_frames[2:] - coeff_frames[1:-1])

    if distance.lower() == "l1":
        return torch.abs(diffs).mean()
    else:
        # default to L2
        return (diffs ** 2).mean()


def plot_coefficient_comparison(
        predicted_coeffs: torch.Tensor,
        target_coeffs: torch.Tensor = None,
        title: str = "Reflection Coefficients",
        save_path: str = "coefficient_trajectories.png",
        show_plot: bool = False
):
    """
    Plot predicted reflection coefficients against target coefficients (if provided).
    For single frame, prints coefficients to console.
    """
    if predicted_coeffs.shape[0] == 1:
        print(f"\n{title} (Single Frame Coefficients):")
        for i, coeff in enumerate(predicted_coeffs.squeeze().numpy(), 1):
            print(f"Coefficient b{i}: {coeff}")

        if target_coeffs is not None and target_coeffs.shape[0] == 1:
            print("\nTarget Coefficients:")
            for i, coeff in enumerate(target_coeffs.squeeze().numpy(), 1):
                print(f"Target b{i}: {coeff}")
        return

    plt.figure(figsize=(10, 6))
    n_coeffs = predicted_coeffs.shape[1]

    if target_coeffs is not None:
        # Plot both predicted and target
        for i in range(n_coeffs):
            plt.plot(target_coeffs[:, i], linestyle='-', label=f"Target b{i + 1}")
            plt.plot(predicted_coeffs[:, i], linestyle='--', label=f"Predicted b{i + 1}")
        plot_title = f"{title} (Target vs. Predicted)"
    else:
        # Plot only predicted coefficients
        for i in range(n_coeffs):
            plt.plot(predicted_coeffs[:, i], label=f"Coefficient b{i + 1}")
        plot_title = title

    plt.title(plot_title)
    plt.xlabel("Frame index")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

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
        b_end: float,
        l_filter_order: int = 2
) -> torch.Tensor:
    """
    Construct mirrored reflection coefficients in a three segment linspace.

    Args:
        n_frames: Number of frames to generate
        b_start: Starting coefficient value
        b_mid: Middle coefficient value
        b_end: Ending coefficient value
        l_filter_order: Number of filter coefficients (default: 2)

    Returns:
        Tensor of shape [n_frames, l_filter_order + 1] with coefficient values
    """
    one_third_frames = n_frames // 3
    two_third_frames = one_third_frames * 2

    # Initialize output tensor - this should be l_filter_order + 1
    coeff_frames = torch.zeros(n_frames, l_filter_order + 1)

    # For each coefficient, create a pattern
    for i in range(l_filter_order + 1):
        if i % 2 == 0:
            # Even indices follow b_start -> b_mid -> b_end pattern
            first_segment = torch.linspace(b_start, b_mid, two_third_frames)
            second_segment = torch.linspace(b_mid, b_end, n_frames - two_third_frames)
            coeff_frames[:, i] = torch.cat([first_segment, second_segment])
        else:
            # Odd indices follow b_end -> b_mid -> b_start pattern (mirrored)
            first_segment = torch.linspace(b_end, b_mid, one_third_frames)
            second_segment = torch.linspace(b_mid, b_start, n_frames - one_third_frames)
            coeff_frames[:, i] = torch.cat([first_segment, second_segment])

    return coeff_frames

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

def plot_upsampled_filter_coeffs(
    model,
    f0_frames: torch.Tensor,
    sample_rate: int,
    length_audio_s: float,
    title: str = "Upsampled Reflection Coefficients",
    save_path: str = "upsampled_filter_coeffs.png",
    show_plot: bool = False
):
    """
    Uses the model's cubic-spline upsampling to generate reflection coefficients
    at the audio sample rate and plots them.

    Args:
        model: The trained DiffKS model (or similar).
        f0_frames: Tensor of fundamental-frequency frames (sample_rate/f0_Hz).
        sample_rate: Sample rate in Hz.
        length_audio_s: Total audio length in seconds.
        title: Title for the plot.
        save_path: Where to save the plotted figure.
        show_plot: Whether to show the figure window instead of saving it.
    """
    n_samples = int(sample_rate * length_audio_s)

    # Evaluate the upsampled delay and filter coefficients
    # for_plotting=True => returns a detached CPU tensor.
    with torch.no_grad():
        up_delay, up_coeffs, _ = model.get_upsampled_parameters(
            delay_len_frames=f0_frames,
            num_samples=n_samples,
            for_plotting=True
        )

    # up_coeffs is shape [n_samples, l_filter_order + 1].
    plt.figure(figsize=(12, 6))
    for i in range(up_coeffs.shape[1]):
        plt.plot(up_coeffs[:, i], label=f"Coeff {i}")

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid(True)

    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

def plot_excitation_filter_analysis(
    burst: torch.Tensor,
    exc_filt_out: torch.Tensor,
    exc_coeffs: torch.Tensor,
    sample_rate: int,
    max_time_s: float = 0.25,
    save_path: str = "excitation_filter_analysis.png",
    show_plot: bool = False
):
    """
    Plots:
      - Top subplot: The first `max_time_s` seconds of:
          (1) The initial noise burst
          (2) The audio after the excitation filter
      - Bottom subplot: The corresponding excitation filter coefficients

    Args:
        burst:          The initial noise burst, shape [burst_size].
        exc_filt_out:   Excitation-filtered audio, shape [n_samples].
        exc_coeffs:     The learned filter coefficients, typically shape [1, burst_size, filter_order]
                        or [burst_size, filter_order].
        sample_rate:    Sample rate of the audio.
        max_time_s:     How many seconds from the start to display for both waveforms.
        save_path:      Where to save the generated figure.
        show_plot:      If True, will display the figure window; otherwise just saves.
    """
    # Convert everything to 1D arrays if necessary
    burst = burst.squeeze()
    exc_filt_out = exc_filt_out.squeeze()

    burst_np = burst.detach().cpu().numpy()
    exc_filt_out_np = exc_filt_out.detach().cpu().numpy()

    # Convert coefficients
    coeffs_np = exc_coeffs.detach().cpu().numpy()
    if coeffs_np.ndim == 3:
        coeffs_np = coeffs_np.squeeze(0)  # shape => [burst_size, filter_order]

    # Slice waveforms to first max_time_s
    burst_len = burst_np.shape[0]
    exc_len = exc_filt_out_np.shape[0]

    max_burst_samps = min(int(sample_rate * max_time_s), burst_len)
    max_exc_samps   = min(int(sample_rate * max_time_s), exc_len)

    burst_np = burst_np[:max_burst_samps]
    burst_time = np.linspace(0, max_time_s, max_burst_samps)

    exc_filt_out_np = exc_filt_out_np[:max_exc_samps]
    exc_time        = np.linspace(0, max_time_s, max_exc_samps)

    if coeffs_np.shape[0] == burst_len:
        coeffs_np = coeffs_np[:max_burst_samps]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle("Excitation Filter Analysis")

    # --- Top subplot: Original burst & excitation-filtered audio
    axs[0].plot(burst_time, burst_np, label="Initial Noise Burst", alpha=0.75)
    axs[0].plot(exc_time, exc_filt_out_np, label="After Excitation Filter", alpha=0.75)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom subplot: Filter coefficients
    for c in range(coeffs_np.shape[1]):
        axs[1].plot(coeffs_np[:, c], label=f"Coeff {c+1}")
    axs[1].set_xlabel("Sample index (of first portion)")
    axs[1].set_ylabel("Filter Coeff Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_excitation_filter_analysis(
    burst: torch.Tensor,
    exc_filt_out: torch.Tensor,
    exc_coeffs: torch.Tensor,
    sample_rate: int,
    max_time_s: float = 0.25,
    save_path: str = "excitation_filter_analysis.png",
    show_plot: bool = False
):
    """
    Plots:
      - Top subplot: The first `max_time_s` seconds of:
          (1) The initial noise burst
          (2) The audio after the excitation filter
      - Bottom subplot: The corresponding excitation filter coefficients

    Args:
        burst:          The initial noise burst, shape [burst_size].
        exc_filt_out:   Excitation-filtered audio, shape [n_samples].
        exc_coeffs:     The learned filter coefficients, typically shape [1, burst_size, filter_order]
                        or [burst_size, filter_order].
        sample_rate:    Sample rate of the audio.
        max_time_s:     How many seconds from the start to display for both waveforms.
        save_path:      Where to save the generated figure.
        show_plot:      If True, will display the figure window; otherwise just saves.
    """
    # Convert everything to 1D arrays if necessary
    burst = burst.squeeze()
    exc_filt_out = exc_filt_out.squeeze()

    burst_np = burst.detach().cpu().numpy()
    exc_filt_out_np = exc_filt_out.detach().cpu().numpy()

    # Convert coefficients
    coeffs_np = exc_coeffs.detach().cpu().numpy()
    if coeffs_np.ndim == 3:
        coeffs_np = coeffs_np.squeeze(0)  # shape => [burst_size, filter_order]

    # Slice waveforms to first max_time_s
    burst_len = burst_np.shape[0]
    exc_len = exc_filt_out_np.shape[0]

    max_burst_samps = min(int(sample_rate * max_time_s), burst_len)
    max_exc_samps   = min(int(sample_rate * max_time_s), exc_len)

    burst_np = burst_np[:max_burst_samps]
    burst_time = np.linspace(0, max_time_s, max_burst_samps)

    exc_filt_out_np = exc_filt_out_np[:max_exc_samps]
    exc_time        = np.linspace(0, max_time_s, max_exc_samps)

    if coeffs_np.shape[0] == burst_len:
        coeffs_np = coeffs_np[:max_burst_samps]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle("Excitation Filter Analysis")

    # --- Top subplot: Original burst & excitation-filtered audio
    axs[0].plot(burst_time, burst_np, label="Initial Noise Burst", alpha=0.75)
    axs[0].plot(exc_time, exc_filt_out_np, label="After Excitation Filter", alpha=0.75)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom subplot: Filter coefficients
    for c in range(coeffs_np.shape[1]):
        axs[1].plot(coeffs_np[:, c], label=f"Coeff {c+1}")
    axs[1].set_xlabel("Sample index (of first portion)")
    axs[1].set_ylabel("Filter Coeff Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_excitation_filter_coefficients(
        model,
        f0_frames: torch.Tensor,
        sample_rate: int,
        length_audio_s: float,
        title: str = "Excitation Filter Coefficients",
        save_path: str = "excitation_coefficients.png",
        show_plot: bool = False
):
    """
    Plots the upsampled excitation filter coefficients over time.

    Args:
        model: The trained DiffKS model
        f0_frames: Tensor of fundamental-frequency frames
        sample_rate: Sample rate in Hz
        length_audio_s: Total audio length in seconds
        title: Title for the plot
        save_path: Where to save the plotted figure
        show_plot: Whether to show the figure window
    """
    n_samples = int(sample_rate * length_audio_s)

    # Get the upsampled parameters
    with torch.no_grad():
        _, _, exc_coeffs = model.get_upsampled_parameters(
            delay_len_frames=f0_frames,
            num_samples=n_samples,
            for_plotting=True
        )

    # Plot the excitation filter coefficients
    plt.figure(figsize=(12, 6))

    # If there's only one coefficient, plot it directly
    if exc_coeffs.shape[1] == 1:
        plt.plot(exc_coeffs.squeeze(), label="Excitation Coefficient")
    else:
        # Plot each coefficient in a different color
        for i in range(exc_coeffs.shape[1]):
            plt.plot(exc_coeffs[:, i], label=f"Coefficient {i + 1}")

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid(True)

    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()