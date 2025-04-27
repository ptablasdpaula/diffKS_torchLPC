from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from utils.helpers import get_device
from .model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
from .preprocess import NsynthDataset
import argparse, os
import multiprocessing as mp
import psutil

import time

from paths import NSYNTH_PREPROCESSED_DIR

def parse_args():
    parser = argparse.ArgumentParser()
    env = os.environ.get

    parser.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-4)))
    parser.add_argument("--batch_size",  type=int, default=int(env("BATCH_SIZE", 16)))
    parser.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 2)))
    parser.add_argument("--hidden_size", type=int, default=int(env("HIDDEN_SIZE", 512)))
    parser.add_argument("--l_order",     type=int, default=int(env("L_ORDER", 5)))
    parser.add_argument("--l_n_frames",  type=int, default=int(env("L_N_FRAMES", 250)))
    parser.add_argument("--exc_order",   type=int, default=int(env("EXC_ORDER", 10)))
    parser.add_argument("--exc_n_frames",type=int, default=int(env("EXC_N_FRAMES", 100)))
    parser.add_argument("--families",    type=str, default=env("FAMILIES", "guitar"))
    parser.add_argument("--sources",     type=str, default=env("SOURCES", "acoustic"))
    parser.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "linear"))
    parser.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    return parser.parse_args()

def main():
    args = parse_args()
    config = {
        "hidden_size": args.hidden_size,
        "loop_order": args.l_order,
        "loop_n_frames": args.l_n_frames,
        "exc_order": args.exc_order,
        "exc_n_frames": args.exc_n_frames,
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": 200,
        "eval_interval": 1,
        "save_dir": "runs/ks_nsynth",
        "families": [f.strip() for f in args.families.split(",")],
        "sources": [s.strip() for s in args.sources.split(",")],
        "num_workers": args.num_workers,
        "interpolation_type": args.interpolation_type,
        "pitch_mode": args.pitch_mode,
    }

    print("\n▶Running with config:")
    for k, v in vars(args).items():
        print(f"   {k:12}: {v}")

    n_samples = 4 * config["sample_rate"]  # 4‑second clips

    print(f"[DEBUG] batch={config['batch_size']}  workers={config['num_workers']}")

    # ─── wandb init ───────────────────────────────────────────── #
    wandb.init(project="diffks-autoencoder", config=config)
    device = get_device()

    print(f"Using device: {device}")

    # ─── overall timing setup ─────────────────────────────────── ❶
    total_iters = 0
    train_start_time = time.time()

    # ─── RAM check ────────────────────────────────────────────── #
    process = psutil.Process(os.getpid())
    print(f"[INFO] Memory at start: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    # ─── Create save directories ───────────────────────────────── #
    full_save_path = os.path.abspath(config["save_dir"])
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Using save directory: {full_save_path}")

    # ─── Data ─────────────────────────────────────────────────── #
    # Updated dataset initialization to use the new implementation
    dataset = NsynthDataset(
        root=NSYNTH_PREPROCESSED_DIR,
        split="test",
        pitch_mode=config["pitch_mode"],
        families=config["families"],
        sources=config["sources"],
    )

    print(f"[INFO] Memory after dataset: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    loader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True if device.type is not "mps" else False,
                        num_workers=config["num_workers"])

    print(f"[INFO] Memory after dataloader: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    # Create model
    model = AE_KarplusModel(
        batch_size=config["batch_size"],
        hidden_size=config["hidden_size"],
        loop_order=config["loop_order"],
        loop_n_frames=config["loop_n_frames"],
        exc_order=config["exc_order"],
        exc_n_frames=config["exc_n_frames"],
        sample_rate=config["sample_rate"],
        interpolation_type=config["interpolation_type"],
        z_encoder=MfccTimeDistributedRnnEncoder(),
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    multi_resolution_stft_loss = MultiResolutionSTFTLoss(scale_invariance=True,
                                                                       perceptual_weighting=True,
                                                                       sample_rate=config["sample_rate"],
                                                                       device=device, )

    # Training loop
    best_loss = float('inf')
    step = 0

    epochs_pbar = tqdm(range(config["num_epochs"]), desc="Training Progress", position=0)

    for epoch in epochs_pbar:
        model.train()
        epoch_loss = 0.0
        batch_count = len(loader)
        total_samples = len(dataset)

        for batch_idx, (audio, pitch, loudness) in enumerate(loader):
            batch_size = audio.shape[0]  # Actual batch size (may be smaller for last batch)
            samples_processed = batch_idx * config["batch_size"] + batch_size

            audio = audio.to(device)
            pitch = pitch.to(device)
            loudness = loudness.to(device)

            epochs_pbar.set_description(
                f"Epoch {epoch + 1}/{config['num_epochs']} [Batch {batch_idx + 1}/{batch_count}, "
                f"Samples {samples_processed}/{total_samples}]")

            #start = time.time()
            output = model(pitch=pitch, loudness=loudness, audio=audio)
            #print(f"[Forward] -> {process.memory_info().rss / 1024 ** 3:.2f}, {time.time() - start:.2f}s")


            #start = time.time()
            loss = multi_resolution_stft_loss(output.unsqueeze(1), audio.unsqueeze(1))
            #print(f"[Loss] -> {process.memory_info().rss / 1024 ** 3:.2f}, {time.time() - start:.2f}s")

            optimizer.zero_grad()

            #start = time.time()
            loss.backward()
            #print(f"[Backward] -> {process.memory_info().rss / 1024 ** 3:.2f}, {time.time() - start:.2f}s")

            optimizer.step()

            # Update progress bar postfix with current loss
            epochs_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Log loss to wandb with step to ensure proper ordering
            wandb.log({"loss": loss.item(), "batch": batch_idx, "step": step})
            epoch_loss += loss.item()
            step += 1
            total_iters += 1

        avg_epoch_loss = epoch_loss / len(loader)

        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch,
            "step": step - 1  # Use the last step from this epoch
        })

        epochs_pbar.set_description(f"Epoch {epoch + 1}/{config['num_epochs']}")
        epochs_pbar.set_postfix(avg_loss=f"{avg_epoch_loss:.4f}")

        if (epoch + 1) % config["eval_interval"] == 0:
            epochs_pbar.write(f"\nEvaluating at epoch {epoch + 1}...")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                model_path = os.path.join(config["save_dir"], 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                epochs_pbar.write(f"Saved best model with loss: {best_loss:.4f} to {model_path}")

            latest_path = os.path.join(config["save_dir"], 'latest_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, latest_path)
            epochs_pbar.write(f"Saved latest checkpoint to {latest_path}")

            # Generate audio sample
            model.eval()
            with torch.no_grad():
                eval_data = next(iter(loader))
                eval_audio, eval_pitch, eval_loudness = [x.to(device) for x in eval_data]
                # No need to normalize loudness here - already done in preprocessing

                hidden = model.get_hidden(eval_pitch, eval_loudness, eval_audio)

                loop_coeffs = model.loop_coeff_proj(hidden).reshape(
                    eval_audio.shape[0], config["loop_n_frames"], model.loop_order + 1)
                exc_coeffs = model.exc_coeff_proj(hidden).reshape(
                    eval_audio.shape[0], config["exc_n_frames"], model.exc_order + 1)

                epochs_pbar.write(f"Generating coefficient plots...")

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
                        title=f"Excitation Coefficients (Epoch {epoch + 1})",
                        stroke="coefficient"
                    ),
                    "epoch": epoch,
                    "step": step
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
                        title=f"Loop Coefficients (Epoch {epoch + 1})",
                        stroke="coefficient"
                    ),
                    "epoch": epoch,
                    "step": step
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
                        title=f"Pitch Over Time (Epoch {epoch + 1})"
                    ),
                    "epoch": epoch,
                    "step": step
                })

                epochs_pbar.write(f"Generating audio samples...")

                # Generate audio
                eval_output = model(eval_pitch, eval_loudness, audio=eval_audio)

                # Save a few examples
                for sample_idx in range(min(3, eval_audio.shape[0])):
                    # Original audio
                    original = eval_audio[sample_idx].cpu().numpy()

                    # Reconstructed audio
                    reconstructed = eval_output[sample_idx].cpu().numpy()

                    # Concatenate original and reconstructed audio
                    comparison = np.concatenate([original, reconstructed])

                    # Save audio file
                    audio_path = os.path.join(config["save_dir"], f'sample_epoch{epoch + 1}_ex{sample_idx}.wav')
                    sf.write(audio_path, comparison, config["sample_rate"])
                    epochs_pbar.write(f"Saved audio sample to {audio_path}")

                    # Log audio to wandb
                    wandb.log({
                        f"audio_example_{sample_idx}": wandb.Audio(
                            comparison,
                            caption=f"Epoch {epoch + 1} Example {sample_idx} (Original + Reconstructed)",
                            sample_rate=config["sample_rate"]
                        ),
                        "epoch": epoch,
                        "step": step
                    })

    # ─── wrap-up timing & wandb finish ──────────────────────────
    total_time = time.time() - train_start_time
    avg_iter_time = total_time / total_iters
    print(f"\n[RESULT] Completed {total_iters} iterations in {total_time:.2f}s → "
          f"avg {avg_iter_time:.3f}s/iter")

    wandb.log({"avg_iter_time": avg_iter_time})

    # Finish wandb run
    wandb.finish()
    print("Training completed!")

if __name__ == '__main__':
    mp.freeze_support()
    main()