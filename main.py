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
    compute_minimum_action, plot_coefficient_comparison, plot_upsampled_filter_coeffs, plot_excitation_filter_analysis,
    plot_excitation_filter_coefficients, save_audio_torchaudio  # <--- import from utils
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
    use_A_weighing     = hp["use_A_weighing"]
    min_action_weight  = hp["min_action_weight"]
    min_action_dist    = hp.get("min_action_distance", "l2")

    patience_epochs = hp["patience_epochs"]
    patience_delta  = hp["patience_delta"]

    exc_order       = mp["exc_order"]
    exc_n_frames    = mp["exc_n_frames"]
    loop_order      = mp["loop_order"]
    loop_n_frames   = mp["loop_n_frames"]
    f0_1_Hz, f0_2_Hz= mp["f0_1_Hz"], mp["f0_2_Hz"]
    f0_n_frames     = mp["f0_n_frames"]
    min_f0_hz       = mp["min_f0_hz"]
    burst_width_s   = mp["burst_width_s"]
    burst_length_s  = mp["burst_length_s"]

    use_double_precision = mp["use_double_precision"]
    normalize_burst = mp["normalize_burst"]
    interp_type = mp["interp_type"]

    use_in_domain = idp["use_in_domain"]
    b_start, b_mid, b_end = idp["b_start"], idp["b_mid"], idp["b_end"]
    t_gain = idp["gain"]

    # ==== Generate Burst ==================================
    burst = noise_burst(
        sample_rate=sample_rate,
        length_s=burst_length_s,
        burst_width_s=burst_width_s,
        normalize=normalize_burst,
    )

    # ==== Create f0 frames (delay lengths) ================
    f0_1_n = sample_rate / f0_1_Hz
    f0_2_n = sample_rate / f0_2_Hz
    f0_frames = torch.linspace(f0_1_n, f0_2_n, f0_n_frames)

    # ==== Initialize Model ================================
    p_model = DiffKS(
        burst=burst,
        loop_n_frames=loop_n_frames,
        sample_rate=sample_rate,
        min_f0_hz=min_f0_hz,
        loop_order=loop_order,
        exc_order=exc_order,
        exc_requires_grad=True,
        interp_type=interp_type,
        use_double_precision=use_double_precision,
        exc_n_frames=exc_n_frames,
    )

    # ==== Create Baseline audio (to be optimized) =========
    _ = ks_to_audio(
        model=p_model,
        out_path="initial.wav",
        f0_frames=f0_frames,
        sample_rate=sample_rate,
        length_audio_s=length_audio_s
    )

    # ==== Generate target audio ===========================
    if use_in_domain:
        t_coeff_frames = make_symmetric_mirrored_coefficient_frame_linspace(
            n_frames=loop_n_frames,
            order=loop_order,
            b_start=b_start,
            b_mid=b_mid,
            b_end=b_end
        )

        t_model = DiffKS(
            burst=burst,
            loop_n_frames=loop_n_frames,
            sample_rate=sample_rate,
            min_f0_hz=min_f0_hz,
            init_loop_b_frames=t_coeff_frames,
            gain=t_gain,
            loop_order=loop_order,
            exc_requires_grad=False,
            interp_type=interp_type,
            use_double_precision=use_double_precision,
        )

        t_audio = ks_to_audio(
            model=t_model,
            out_path="target.wav",
            f0_frames=f0_frames,
            sample_rate=sample_rate,
            length_audio_s=length_audio_s
        )
    else:
        t_audio = process_target(
            target_path="audio/guitar.wav",
            out_path="audio/out/target.wav",
            target_sample_rate=sample_rate,
            target_length_samples=length_audio_n
        )

    # ==== Setup optimizer and loss ========================
    optimizer = torch.optim.Adam(p_model.parameters(), lr=learning_rate)
    loss_fn = multi_stft_loss(scale_invariance=True,
                              perceptual_weighting=use_A_weighing,
                              sample_rate=sample_rate,)

    # For plotting the loss curve
    loss_curve = []

    bad_epochs = 0 # For Early Stopping

    progress_bar = tqdm(range(max_epochs), desc="Training")
    for epoch in progress_bar:
        # Forward
        output = p_model(delay_len_frames=f0_frames,
                         n_samples=length_audio_n,
                         target=t_audio if use_in_domain is False else None,)

        # Multi-resolution STFT loss
        stft_loss = loss_fn(
            output.unsqueeze(0).unsqueeze(0),
            t_audio.unsqueeze(0).unsqueeze(0)
        ) * stft_weight

        loss = stft_loss

        if loop_n_frames > 1:
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

        if loop_n_frames > 1:
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
    plt.savefig("plots/loss_curve.png")  # or plt.show()
    plt.close()

    # ==== Plot predicted vs target reflection coeffs ======
    with torch.no_grad():
        pred_coeffs = p_model.get_constrained_coefficients(for_plotting=True)
        target_coeffs = t_model.get_constrained_coefficients(for_plotting=True) if use_in_domain else None

        plot_coefficient_comparison(
            predicted_coeffs=pred_coeffs,
            target_coeffs=target_coeffs,
            save_path="plots/coefficient_trajectories.png"
        )

        print (f"The Predicted Gain is {p_model.get_gain()}")
        if use_in_domain: print(f"The Target Gain is {t_model.get_gain()}")

    # ==== Plot predicted FINAL coefficients (after upsampling)
    plot_upsampled_filter_coeffs(
        model=p_model,
        f0_frames=f0_frames,
        sample_rate=sample_rate,
        length_audio_s=length_audio_s,
        title="Cubic predicted",
        save_path="plots/coefficient_upsampled.png"
    )

    # ==== Excitation Filter Analysis =============================
    with torch.no_grad():
        exc_filt_out = p_model.get_excitation_filter_out()
        exc_coeffs = p_model.exc_coefficients
        burst_in = p_model.excitation

    plot_excitation_filter_analysis(
        burst=burst_in,
        exc_filt_out=exc_filt_out,
        exc_coeffs=exc_coeffs,
        sample_rate=sample_rate,
        max_time_s=burst_length_s,
        save_path="plots/excitation_filter_analysis.png",
        show_plot=False
    )

    # ==== Plot Excitation Filter coefficients after upsampling ==
    plot_excitation_filter_coefficients(
        model=p_model,
        f0_frames=f0_frames,
        sample_rate=sample_rate,
        length_audio_s=length_audio_s,
        title="Excitation Filter Coefficients",
        save_path="plots/excitation_coefficients.png"
    )

    # ==== Plot inverse filtered signal (plucking) =========
    inv_filt_signal = p_model.get_inverse_filtered_signal()
    time_axis = torch.arange(len(inv_filt_signal)) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, inv_filt_signal.detach().numpy())
    plt.title("Inverse Filtered Signal (Plucking Excitation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/inverse_filtered_signal.png")  # or use plt.show() to display it
    plt.close()

    save_audio_torchaudio(inv_filt_signal, sample_rate=sample_rate, out_path="audio/out/inversed.wav")

    # ==== Save final model output =========================
    with torch.no_grad():
        ks_to_audio(
            model=p_model,
            out_path="optimized_model.wav",
            f0_frames=f0_frames,
            sample_rate=sample_rate,
            length_audio_s=length_audio_s,
            target_audio=t_audio,
        )

if __name__ == "__main__":
    main()