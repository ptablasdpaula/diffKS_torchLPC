import torch
from auraloss.freq import MultiResolutionSTFTLoss as multi_stft_loss
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffKS import DiffKS, noise_burst
from utils import (
    load_config,
    make_symmetric_mirrored_coefficient_frame_linspace,
    ks_to_audio,
    process_target,
    compute_minimum_action, plot_coefficient_comparison  # <--- import from utils
)

def main():
    # ==== Retrieve from config ============================
    hp, mp, idp, gs = load_config()

    sample_rate    = gs["sample_rate"]
    length_audio_s = gs["length_audio_s"]
    length_audio_n = sample_rate * length_audio_s

    learning_rate      = hp["learning_rate"]
    max_epochs         = hp["max_epochs"]
    stft_weight        = hp["stft_weight"]
    min_action_weight  = hp["min_action_weight"]
    min_action_dist    = hp.get("min_action_distance", "l2")

    patience_epochs = hp["patience_epochs"]
    patience_delta = hp["patience_delta"]

    f0_1_Hz, f0_2_Hz   = mp["f0_1_Hz"], mp["f0_2_Hz"]
    n_frames           = mp["n_frames"]
    burst_width_s      = mp["burst_width_in_s"]
    lowest_note_in_hz  = mp["lowest_note_in_hz"]
    loop_filter_order = mp["l_filter_order"]
    exc_filter_order = mp["exc_filter_order"]

    use_in_domain      = idp["use_in_domain"]
    b_start, b_mid, b_end = idp["b_start"], idp["b_mid"], idp["b_end"]
    t_gain = idp["gain"]

    # ==== Generate Burst ==================================
    burst = noise_burst(
        sample_rate=sample_rate,
        length_s=length_audio_s,
        burst_width_s=burst_width_s
    )

    # ==== Create f0 frames (delay lengths) ================
    f0_1_n = sample_rate / f0_1_Hz
    f0_2_n = sample_rate / f0_2_Hz
    f0_frames = torch.linspace(f0_1_n, f0_2_n, n_frames)

    # ==== Initialize Model ================================
    p_model = DiffKS(
        burst=burst,
        n_frames=n_frames,
        sample_rate=sample_rate,
        lowest_note_in_hz=lowest_note_in_hz,
        l_filter_order=loop_filter_order,
        excitation_filter_order=exc_filter_order,
        requires_grad=True,
    )

    # ==== Create Baseline audio (to be optimized) =========
    _ = ks_to_audio(
        model=p_model,
        out_path="audio/initial.wav",
        f0_frames=f0_frames,
        sample_rate=sample_rate,
        length_audio_s=length_audio_s
    )

    # ==== Generate target audio ===========================
    if use_in_domain:
        t_coeff_frames = make_symmetric_mirrored_coefficient_frame_linspace(
            n_frames=n_frames,
            l_filter_order=loop_filter_order,
            b_start=b_start,
            b_mid=b_mid,
            b_end=b_end
        )

        t_model = DiffKS(
            burst=burst,
            n_frames=n_frames,
            sample_rate=sample_rate,
            lowest_note_in_hz=lowest_note_in_hz,
            init_coeffs_frames=t_coeff_frames,
            gain=t_gain,
            l_filter_order=loop_filter_order,
            requires_grad=False
        )

        t_audio = ks_to_audio(
            model=t_model,
            out_path="audio/target.wav",
            f0_frames=f0_frames,
            sample_rate=sample_rate,
            length_audio_s=length_audio_s
        )
    else:
        t_audio = process_target(
            target_path="audio/guitar.wav",
            out_path="audio/target.wav",
            target_sample_rate=sample_rate,
            target_length_samples=length_audio_n
        )

    # ==== Setup optimizer and loss ========================
    optimizer = torch.optim.Adam(p_model.parameters(), lr=learning_rate)
    loss_fn = multi_stft_loss(scale_invariance=True)

    # For plotting the loss curve
    loss_curve = []

    bad_epochs = 0 # For Early Stopping

    progress_bar = tqdm(range(max_epochs), desc="Training")
    for epoch in progress_bar:
        # Forward
        output = p_model(delay_len_frames=f0_frames, n_samples=length_audio_n)

        # Multi-resolution STFT loss
        stft_loss = loss_fn(
            output.unsqueeze(0).unsqueeze(0),
            t_audio.unsqueeze(0).unsqueeze(0)
        ) * stft_weight

        loss = stft_loss

        if n_frames > 1:
            # Minimum-action loss on raw coeff frames
            min_action_loss = compute_minimum_action(
                p_model.raw_coeff_frames,
                distance=min_action_dist
            ) * min_action_weight

            # Combine them with separate weights
            loss += min_action_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track and display
        loss_curve.append(loss.item())

        if n_frames > 1:
            progress_bar.set_postfix({
                "stft_loss": f"{stft_loss.item():.4f}",
                "ma_loss": f"{min_action_loss.item():.4f}",
                "total": f"{loss.item():.4f}"
            })
        else:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if loss.item() < patience_delta:
            bad_epochs += 1
        else:
            bad_epochs = 0

        if bad_epochs >= patience_epochs:
            print(f"No improvement after {patience_epochs} epochs. Early stopping at epoch {epoch}.")
            break

    print("Training finished")

    # ==== Plot final loss curve ===========================
    plt.figure()
    plt.plot(loss_curve, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # or plt.show()
    plt.close()

    # ==== Plot predicted vs target reflection coeffs ======
    with torch.no_grad():
        pred_coeffs = p_model.get_constrained_coefficients(for_plotting=True)
        target_coeffs = t_model.get_constrained_coefficients(for_plotting=True) if use_in_domain else None

        plot_coefficient_comparison(
            predicted_coeffs=pred_coeffs,
            target_coeffs=target_coeffs,
            save_path="coefficient_trajectories.png"
        )

        print (f"The Predicted Gain is {p_model.get_gain()}")
        if use_in_domain: print(f"The Target Gain is {t_model.get_gain()}")

    # ==== Save final model output =========================
    with torch.no_grad():
        ks_to_audio(
            model=p_model,
            out_path="audio/optimized_model.wav",
            f0_frames=f0_frames,
            sample_rate=sample_rate,
            length_audio_s=length_audio_s
        )

if __name__ == "__main__":
    main()