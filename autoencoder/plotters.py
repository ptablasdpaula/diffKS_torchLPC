import numpy as np

def plot_composite_four(fig_path: str,
                       target: np.ndarray,
                       reconstructed: np.ndarray,
                       loop_coeffs_c: np.ndarray,
                       eq_gains: np.ndarray,
                       sr: int) -> None:
    """
    Create a 4-panel composite:
      1) Target waveform
      2) Reconstructed waveform
      3) Loop filter coefficients
      4) EQ Band Gains (per-band dB)
    Saves to `fig_path`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes = axes.ravel()

    # 1) Target waveform
    ax = axes[0]
    t = np.arange(len(target)) / sr
    ax.plot(t, target)
    ax.set_title("Target")
    ax.set_xlabel("Time (s)")

    # 2) Reconstructed waveform
    ax = axes[1]
    t_rec = np.arange(len(reconstructed)) / sr
    ax.plot(t_rec, reconstructed)
    ax.set_title("Reconstructed")
    ax.set_xlabel("Time (s)")

    # 3) Loop filter coefficients
    ax = axes[2]
    if loop_coeffs_c is not None:
        for k in range(loop_coeffs_c.shape[1]):
            ax.plot(np.arange(loop_coeffs_c.shape[0]), loop_coeffs_c[:, k], label=f"b{k}")
        ax.set_title("Loop Coefficients")
        ax.set_xlabel("Tap Index")
        ax.set_ylabel("Coefficient")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(loop coeffs unavailable)", ha="center", va="center")
        ax.set_title("Loop Coefficients")

    # 4) EQ Band Gains
    ax = axes[3]
    ax.set_title("EQ Band Gains")
    if eq_gains is not None and len(eq_gains) > 0:
        band_indices = np.arange(1, len(eq_gains)+1)
        ax.plot(band_indices, eq_gains, marker='o', linestyle='-')
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Gain (dB)")
        ax.set_ylim([-12, 12])
        ax.set_xticks(band_indices)
        ax.grid(True, which="both", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "(no EQ gains)", ha="center", va="center")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Gain (dB)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# === New helper functions for zoomed plotting of excitation and gain signals ===
def _resample_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1D array x to target_len samples."""
    if x is None:
        return None
    if x.ndim != 1:
        x = np.ravel(x)
    n = len(x)
    if n == target_len or n == 0:
        return x.copy()
    xp = np.linspace(0, n - 1, num=n)
    fp = x
    x_new = np.linspace(0, n - 1, num=target_len)
    return np.interp(x_new, xp, fp)


def _find_zoom_windows(trigger_signal: np.ndarray, sr: int, pre_ms: float = 10.0, post_ms: float = 10.0, max_windows: int = 5):
    """
    Find up to `max_windows` windows around rising edges where trigger_signal != 0.
    Returns a list of (start_idx, end_idx) sample index tuples.
    If no non-zero region exists, return a single centered window of 40 ms.
    """
    if trigger_signal.ndim != 1:
        trigger_signal = np.ravel(trigger_signal)
    n = len(trigger_signal)
    pre_samp = int(round(pre_ms * 1e-3 * sr))
    post_samp = int(round(post_ms * 1e-3 * sr))

    nz = np.abs(trigger_signal) > 0
    edges = np.flatnonzero((~nz[:-1]) & (nz[1:])) + 1 if n > 1 else np.array([], dtype=int)

    windows = []
    if edges.size == 0:
        # Fallback: centered 40 ms window
        half = int(round((pre_samp + post_samp)))
        c = n // 2
        s = max(0, c - half)
        e = min(n, c + half)
        windows.append((s, e))
        return windows

    for idx in edges[:max_windows]:
        s = max(0, idx - pre_samp)
        e = min(n, idx + post_samp)
        # Avoid appending empty or degenerate windows
        if e > s:
            windows.append((s, e))
    return windows


def plot_excitation_zoomed(fig_path: str,
                            sr: int,
                            excitation_pregain: np.ndarray,
                            gain_frames: np.ndarray,
                            gain_up: np.ndarray,
                            pre_ms: float = 10.0,
                            post_ms: float = 10.0,
                            max_windows: int = 5) -> None:
    """
    Create a figure zooming around points where `excitation_pregain != 0`.
    Overlays three signals in each zoom window: excitation_pregain, gain_frames (resampled), and gain_up.
    Saves to `fig_path`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if excitation_pregain.ndim != 1:
        excitation_pregain = np.ravel(excitation_pregain)
    if gain_up is not None and gain_up.ndim != 1:
        gain_up = np.ravel(gain_up)

    # Ensure we can overlay gain_frames by resampling it to audio rate
    gf_res = None
    try:
        if gain_frames is not None:
            if gain_frames.ndim != 1:
                gain_frames = np.ravel(gain_frames)
            gf_res = _resample_to_len(gain_frames, len(excitation_pregain))
        if gain_up is not None and len(gain_up) != len(excitation_pregain):
            gain_up = _resample_to_len(gain_up, len(excitation_pregain))
    except Exception:
        gf_res = None

    windows = _find_zoom_windows(excitation_pregain, sr, pre_ms=pre_ms, post_ms=post_ms, max_windows=max_windows)

    n_panels = max(1, len(windows))
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    t = np.arange(len(excitation_pregain)) / sr

    for ax, (s, e) in zip(axes, windows):
        ax.plot(t[s:e], excitation_pregain[s:e], label="excitation_pregain")
        if gf_res is not None:
            ax.plot(t[s:e], gf_res[s:e], label="gain_frames(resampled)")
        if gain_up is not None:
            ax.plot(t[s:e], gain_up[s:e], label="gain_up")
        ax.set_xlim(t[s], t[e - 1] if e - 1 < len(t) else t[-1])
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Zoom: excitation_pregain, gain_frames, gain_up", y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_signal_zoomed(fig_path: str,
                       sr: int,
                       signal: np.ndarray,
                       reference_for_windows: np.ndarray,
                       title: str,
                       pre_ms: float = 10.0,
                       post_ms: float = 10.0,
                       max_windows: int = 5) -> None:
    """
    Plot a single signal using zoom windows computed from `reference_for_windows` (e.g., excitation_pregain).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if signal.ndim != 1:
        signal = np.ravel(signal)
    if reference_for_windows.ndim != 1:
        reference_for_windows = np.ravel(reference_for_windows)

    windows = _find_zoom_windows(reference_for_windows, sr, pre_ms=pre_ms, post_ms=post_ms, max_windows=max_windows)

    n_panels = max(1, len(windows))
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    t = np.arange(len(signal)) / sr

    for ax, (s, e) in zip(axes, windows):
        ax.plot(t[s:e], signal[s:e])
        ax.set_xlim(t[s], t[e - 1] if e - 1 < len(t) else t[-1])
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Time (s)")

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# === Composite excitation plot: 3 stacked panels, zoomed on first trigger ===
def plot_excitation_composite_zoomed(fig_path: str,
                                     sr: int,
                                     excitation_pregain: np.ndarray,
                                     excitation_postgain: np.ndarray,
                                     excitation: np.ndarray,
                                     gain_frames: np.ndarray,
                                     gain_up: np.ndarray,
                                     pre_ms: float = 10.0,
                                     post_ms: float = 10.0) -> None:
    """
    Single composite figure with three stacked panels (zoomed on first trigger):
      1) excitation_pregain overlaid with gain_frames (resampled) and gain_up
      2) excitation_postgain
      3) excitation (final)
    Asserts expected lengths/relations when duration is ~4s.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten to 1D
    pre  = np.ravel(excitation_pregain)
    post = np.ravel(excitation_postgain)
    fin  = np.ravel(excitation)
    gf   = None if gain_frames is None else np.ravel(gain_frames)
    gu   = None if gain_up     is None else np.ravel(gain_up)

    # Basic consistency checks
    assert len(pre) == len(post) == len(fin), (
        f"Excitation lengths must match: pre={len(pre)}, post={len(post)}, final={len(fin)}")
    if gu is not None:
        assert len(gu) == len(pre), (
            f"gain_up length must match audio-rate signals: gain_up={len(gu)} vs pre={len(pre)}")

    seconds = len(pre) / float(sr)
    # Resample frame-rate gains to audio rate for overlay
    gf_res = None
    if gf is not None:
        gf_res = _resample_to_len(gf, len(pre))

    # Determine zoom window: first rising edge of pre-gain excitation (fallback: centered window)
    windows = _find_zoom_windows(pre, sr, pre_ms=pre_ms, post_ms=post_ms, max_windows=1)
    s, e = windows[0]

    t = np.arange(len(pre)) / float(sr)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    # Top: pre + gains
    ax0 = axes[0]
    ax0.plot(t[s:e], pre[s:e], label="excitation_pregain")
    if gf_res is not None:
        ax0.plot(t[s:e], gf_res[s:e], label="gain_frames (resampled)")
    if gu is not None:
        ax0.plot(t[s:e], gu[s:e], label="gain_up")
    ax0.set_ylabel("amp")
    ax0.set_title("Excitation pre-gain + gains (zoom)")
    ax0.grid(True, alpha=0.2)
    ax0.legend(fontsize=8, loc="upper right")

    # Middle: post-gain
    ax1 = axes[1]
    ax1.plot(t[s:e], post[s:e])
    ax1.set_ylabel("amp")
    ax1.set_title("Excitation post-gain (zoom)")
    ax1.grid(True, alpha=0.2)

    # Bottom: final excitation
    ax2 = axes[2]
    ax2.plot(t[s:e], fin[s:e])
    ax2.set_ylabel("amp")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Excitation final (zoom)")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)