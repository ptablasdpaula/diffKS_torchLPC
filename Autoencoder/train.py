from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, os, wandb, auraloss
from torch.utils.data import DataLoader
from utils import get_device
from model import AE_KarplusModel
from preprocess import NsynthDataset
import multiprocessing as mp

from ddsp_pytorch.ddsp.core import mean_std_loudness
from paths import NSYNTH_DIR

# Configuration
config = {
    "hidden_size":     512,
    "loop_order":      10,
    "loop_n_frames":   250,
    "exc_order":       10,
    "exc_n_frames":    100,
    "sample_rate":     16000,
    "batch_size":      8,
    "learning_rate":   1e-3,
    "num_epochs":      300,
    "eval_interval":   50,
    "save_dir":        "runs/ks_nsynth",
    # dataset options
    "split":           "test",                 # train / valid / test
    "families":        ["guitar"],
    "sources":         ["acoustic"],
}

n_samples = 4 * config["sample_rate"]          # 4‑second clips

def main():
    # ─── wandb init ───────────────────────────────────────────── #
    wandb.init(project="diffks-autoencoder", config=config)
    device = get_device()

    # ─── Data ─────────────────────────────────────────────────── #
    dataset = NsynthDataset(
        root_dir    = NSYNTH_DIR,
        split       = config["split"],
        families    = set(config["families"]),
        sources     = set(config["sources"]),
        sample_rate = config["sample_rate"],
        hop_size    = 256,
        segment_length = n_samples,
    )

    loader = DataLoader(dataset,
                        batch_size  = config["batch_size"],
                        shuffle     = True,
                        drop_last   = True,
                        pin_memory  = True,
                        num_workers = 4)

    # Calculate mean and std of loudness for normalization
    mean_loudness, std_loudness = mean_std_loudness(dataset)

    # Create model
    model = AE_KarplusModel(
        batch_size      = config["batch_size"],
        hidden_size     = config["hidden_size"],
        loop_order      = config["loop_order"],
        loop_n_frames   = config["loop_n_frames"],
        exc_order       = config["exc_order"],
        exc_n_frames    = config["exc_n_frames"],
        sample_rate     = config["sample_rate"],
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Create multi-resolution spectral loss using auraloss
    multi_resolution_stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        win_lengths=[2048, 1024, 512, 256, 128, 64],
        w_sc=0.5,  # Spectral convergence loss weight
        w_log_mag=0.5,  # Log magnitude loss weight
        w_lin_mag=0.5,  # Linear magnitude loss weight
        w_phs=0.0  # Phase loss weight (disabled)
    ).to(device)

    # Training loop
    best_loss = float('inf')
    step = 0

    for epoch in range(config["num_epochs"]):
        print(f"\n{'=' * 30}\nStarting Epoch {epoch + 1}/{config['num_epochs']}\n{'=' * 30}")
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for batch_idx, (audio, pitch, loudness) in enumerate(progress_bar):
            audio = audio.to(device)
            pitch = pitch.to(device)
            loudness = loudness.to(device)

            loudness = (loudness - mean_loudness) / (std_loudness + 1e-6)

            output = model(pitch=pitch,
                           loudness=loudness,
                           input=audio)

            loss = multi_resolution_stft_loss(output.unsqueeze(1), audio.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Log loss to wandb
            wandb.log({"loss": loss.item(), "step": step})
            epoch_loss += loss.item()
            step += 1

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(loader)
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})

        # Evaluate and save model periodically
        if (epoch + 1) % config["eval_interval"] == 0:
            # Save model if it's the best so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), os.path.join(config["save_dir"], 'best_model.pth'))
                print(f"Saved best model with loss: {best_loss:.4f}")

            # Save latest model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(config["save_dir"], 'latest_model.pth'))

            # Generate audio sample
            model.eval()
            with torch.no_grad():
                # Get first batch from dataloader for consistent evaluation
                eval_data = next(iter(loader))
                eval_audio, eval_pitch, eval_loudness = [x.to(device) for x in eval_data]
                eval_loudness = (eval_loudness - mean_loudness) / std_loudness

                hidden = model.get_hidden(eval_pitch, eval_loudness)

                loop_coeffs = model.loop_coeff_proj(hidden).reshape(
                    config["batch_size"], config["loop_n_frames"], model.loop_order + 1)
                exc_coeffs = model.exc_coeff_proj(hidden).reshape(
                    config["batch_size"], config["exc_n_frames"], model.exc_order + 1)

                # For excitation coefficients
                exc_data = []
                for frame_idx in range(config["exc_n_frames"]):
                    for coeff_idx in range(model.exc_order + 1):
                        exc_data.append([
                            frame_idx,
                            exc_coeffs[0, frame_idx, coeff_idx].item(),
                            f"Coefficient {coeff_idx}"
                        ])

                exc_table = wandb.Table(data=exc_data, columns=["frame", "value", "coefficient"])
                wandb.log({
                    "all_excitation_coefficients": wandb.plot.line(
                        exc_table,
                        "frame",
                        "value",
                        title="All Excitation Coefficients",
                        stroke="coefficient"
                    ),
                    "epoch": epoch
                })

                # For loop coefficients
                loop_data = []
                for frame_idx in range(config["loop_n_frames"]):
                    for coeff_idx in range(model.loop_order + 1):
                        loop_data.append([
                            frame_idx,
                            loop_coeffs[0, frame_idx, coeff_idx].item(),
                            f"Coefficient {coeff_idx}"
                        ])

                loop_table = wandb.Table(data=loop_data, columns=["frame", "value", "coefficient"])
                wandb.log({
                    "all_loop_coefficients": wandb.plot.line(
                        loop_table,
                        "frame",
                        "value",
                        title="All Loop Coefficients",
                        stroke="coefficient"
                    ),
                    "epoch": epoch
                })

                # For pitch
                pitch_data = []
                for frame_idx in range(len(eval_pitch[0])):
                    pitch_data.append([
                        frame_idx,
                        eval_pitch[0, frame_idx, 0].item()
                    ])

                pitch_table = wandb.Table(data=pitch_data, columns=["frame", "value"])
                wandb.log({
                    "pitch_over_time": wandb.plot.line(
                        pitch_table,
                        "frame",
                        "value",
                        title="Pitch Over Time"
                    ),
                    "epoch": epoch
                })

                # Generate audio
                eval_output = model(eval_pitch, eval_loudness, input=eval_audio)

                # Save a few examples
                for sample_idx in range(min(3, config["batch_size"])):
                    # Original audio
                    original = eval_audio[sample_idx].cpu().numpy()

                    # Reconstructed audio
                    reconstructed = eval_output[sample_idx].cpu().numpy()

                    # Concatenate original and reconstructed audio
                    comparison = np.concatenate([original, reconstructed])

                    # Save audio file
                    sf.write(
                        os.path.join(config["save_dir"], f'sample_epoch{epoch + 1}_ex{sample_idx}.wav'),
                        comparison,
                        config["sample_rate"]
                    )

                    # Log audio to wandb
                    wandb.log({
                        f"audio_example_{sample_idx}": wandb.Audio(
                            comparison,
                            caption=f"Epoch {epoch + 1} Example {sample_idx} (Original + Reconstructed)",
                            sample_rate=config["sample_rate"]
                        ),
                        "epoch": epoch
                    })

    # Finish wandb run
    wandb.finish()
    print("Training completed!")

if __name__ == '__main__':
    mp.freeze_support()
    main()