from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from .model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
import argparse, os, json
import multiprocessing as mp
import psutil
import math

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset, a_weighted_loudness
from data.synthetic_generate import random_param_batch
from utils import get_device, str2bool

def parse_args():
    p = argparse.ArgumentParser()
    env = os.environ.get

    p.add_argument("--name", type=str, default=env("NAME", "exp"),
                        help="Unique experiment name (used for checkpoints & wandb run)")
    p.add_argument("--continue_from_checkpoint", action="store_true",
                        default=str2bool(env("CONTINUE_FROM_CHECKPOINT", "false")),
                        help="Resume training from latest checkpoint for this --name")

    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-4)))

    p.add_argument("--batch_size",  type=int, default=int(env("BATCH_SIZE", 8)))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 2)))

    p.add_argument("--hidden_size", type=int, default=int(env("HIDDEN_SIZE", 512)))

    p.add_argument("--l_order",     type=int, default=int(env("L_ORDER", 2)))
    p.add_argument("--l_n_frames",  type=int, default=int(env("L_N_FRAMES", 16)))
    p.add_argument("--exc_order",   type=int, default=int(env("EXC_ORDER", 5)))
    p.add_argument("--exc_n_frames",type=int, default=int(env("EXC_N_FRAMES", 25)))

    p.add_argument("--families",    type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources",     type=str, default=env("SOURCES", "acoustic"))

    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "linear"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))
    p.add_argument("--loudness_loss_delta", type=float, default=float(env("LOUDNESS_LOSS_DELTA", 0)))

    p.add_argument(
        "--parameter_loss",
        action="store_true",
        default=str2bool(env("PARAMETER_LOSS", "false")),
        help="Use parameter-loss training (overrides default if PARAMETER_LOSS env is set)"
    )

    p.add_argument("--batches_per_epoch", type=int, default=int(env("BATCHES_PER_EPOCH", 100)))

    return p.parse_args()

def main():
    args = parse_args()
    config = {
        "hidden_size": args.hidden_size,
        "loop_order": args.l_order,
        "loop_n_frames": args.l_n_frames,
        "exc_order": args.exc_order,
        "exc_n_frames": args.exc_n_frames,
        "sample_rate": 16000,
        "ks_sample_rate": 41000,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": 200,
        "eval_interval": 1,
        "save_dir": f"autoencoder/runs/{args.name}",
        "families": [f.strip() for f in args.families.split(",")],
        "sources": [s.strip() for s in args.sources.split(",")],
        "num_workers": args.num_workers,
        "interpolation_type": args.interpolation_type,
        "pitch_mode": args.pitch_mode,
        "loudness_loss_delta": args.loudness_loss_delta,
        "parameter_loss": args.parameter_loss,
        "batches_per_epoch": args.batches_per_epoch,
    }

    print("\n▶Running with config:")
    for k, v in vars(args).items():
        print(f"   {k:12}: {v}")

    # ─── device init ───────────────────────────────────────────────────────
    device = get_device()
    print(f"Using device: {device}")

    # ─── WandB init ────────────────────────────────────────────────────────
    wandb_id = None
    latest_ckpt = os.path.join(config["save_dir"], f"latest_model_{args.name}.pth")
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        tmp_ckpt = torch.load(latest_ckpt, map_location="cpu")
        wandb_id = tmp_ckpt.get("wandb_id", None)
        print(f"[INFO] Found checkpoint – will resume run id {wandb_id}")

    autoencoder_dir = os.path.dirname(os.path.abspath(__file__))
    wandb_run = wandb.init(project="diffks-autoencoder", name=args.name, dir=autoencoder_dir,
                           id=wandb_id, resume="allow", config=config)
    if wandb_id is None:
        wandb_id = wandb_run.id  # store for fresh runs

    def log_batch(metric_name, value):
        if wandb.run is not None:
            wandb.log({metric_name: value})

    # ─── RAM check ────────────────────────────────────────────── #
    process = psutil.Process(os.getpid())
    print(f"[INFO] Memory at start: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    # ─── Create save directories ───────────────────────────────── #
    full_save_path = os.path.abspath(config["save_dir"])
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Using save directory: {full_save_path}")

    # ─── Data ─────────────────────────────────────────────────────────────
    if not config["parameter_loss"]:
        dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                                split="train",
                                pitch_mode=config["pitch_mode"],
                                families=config["families"],
                                sources=config["sources"], )

        train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                                  drop_last=True, pin_memory=True if device.type != "mps" else False,
                                  num_workers=config["num_workers"])

        train_gen = None
    else:
        dataset = None
        train_loader = None
        train_gen = torch.Generator(device=device).manual_seed(42)


    val_dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                                split="test",
                                pitch_mode=config["pitch_mode"],
                                families=config["families"],
                                sources=config["sources"], )

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            drop_last=True, pin_memory=True if device.type != "mps" else False, num_workers=config["num_workers"])

    # ─── Start Model, optimizer & Loss ────────────────────────── #
    model = AE_KarplusModel(batch_size=config["batch_size"], hidden_size=config["hidden_size"],
                            loop_order=config["loop_order"], loop_n_frames=config["loop_n_frames"],
                            exc_order=config["exc_order"], exc_n_frames=config["exc_n_frames"],
                            internal_sr=config["ks_sample_rate"], interpolation_type=config["interpolation_type"],
                            z_encoder=MfccTimeDistributedRnnEncoder(),).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    mr_stft = MultiResolutionSTFTLoss(scale_invariance=True, perceptual_weighting=True,
                                      sample_rate=config["sample_rate"], device=device, )

    # ─── Resume from checkpoint if requested ──────────────────────────────
    start_epoch, best_val_loss = 0, float('inf')
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)

        # strip out the two analysis buffers so their old shapes don't conflict
        sd = ckpt["model_state_dict"]
        for key in list(sd):
            if "ks_inverse_signal" in key or "excitation_filter_out" in key:
                sd.pop(key)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

        print(f"[RESUME] Starting at epoch {start_epoch} (best so far {best_val_loss:.4f})")

    bpe = config["batches_per_epoch"] if config["parameter_loss"] else len(train_loader)

    # ───────────────────────── training epochs ───────────────────────────
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        t_loss = 0

        if config["parameter_loss"]:
            for i in tqdm(range(bpe), desc=f"[E{epoch:03d} train]"):
                # Generate random parameters
                audio, pitch, loud, loop_coeffs, exc_coeffs = random_param_batch(
                    model.decoder, config["batch_size"], generator=train_gen,
                )

                audio = audio.to(dtype=torch.float32, device=device)
                pitch = pitch.to(dtype=torch.float32, device=device)
                loud = loud.to(dtype=torch.float32, device=device)

                # Generate random audio
                pred_loop_coeffs, pred_exc_coeffs = model(
                    pitch=pitch, loudness=loud,
                    audio=audio, audio_sr=config["sample_rate"],
                    return_parameters=True
                )

                loss = torch.nn.functional.l1_loss(pred_loop_coeffs, loop_coeffs)
                loss += torch.nn.functional.l1_loss(pred_exc_coeffs, exc_coeffs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Per-Batch train loss
                log_batch("train loss per batch", loss.item())

                t_loss += loss.item()
        else:
            for audio, pitch, loud in tqdm(train_loader, desc=f"[E{epoch:03d} train]"):
                audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                recon = model(pitch=pitch, loudness=loud, audio=audio, audio_sr=config["sample_rate"])

                loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Per batch train loss
                log_batch("train loss per batch", loss.item())
                t_loss += loss.item()

        t_loss /= bpe

        # ░░ VALID ░░ (every eval_interval)
        if epoch % config["eval_interval"] == 0:
            if config["parameter_loss"]:
                # validate on "synthetic" param batches
                model.eval()
                v_losses = []

                # Number of synthetic batches to cover (approx. 200 items)
                n_val_batches = math.ceil(200 / config["batch_size"])

                with torch.no_grad():
                    for _ in range(n_val_batches):
                        audio, pitch, loud, loop_c, exc_c = random_param_batch(
                            model.decoder, config["batch_size"], generator=train_gen,
                        )

                        audio = audio.to(dtype=torch.float32, device=device)
                        pitch = pitch.to(dtype=torch.float32, device=device)
                        loud = loud.to(dtype=torch.float32, device=device)

                        pred_loop, pred_exc = model(pitch=pitch,
                                                    loudness=loud,
                                                    audio=audio,
                                                    audio_sr=config["sample_rate"],
                                                    return_parameters=True)
                        batch_v = (
                            torch.nn.functional.l1_loss(pred_loop, loop_c).item() +
                            torch.nn.functional.l1_loss(pred_exc, exc_c).item()
                        )

                        # Compute and log per-batch validation loss
                        log_batch("val loss per batch", batch_v)
                        v_losses.append(batch_v)

                v_loss = float(np.mean(v_losses))

            else:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for audio, pitch, loud in val_loader:
                        audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                        recon = model(pitch=pitch, loudness=loud, audio=audio, audio_sr=config["sample_rate"])

                        # compute and log per-batch val loss
                        batch_v = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)).item()
                        log_batch("val loss per batch", batch_v)
                        v_losses.append(batch_v)
                v_loss = float(np.mean(v_losses))
        else:
            v_loss = np.nan

        # ░░ logging ░░
        log_batch("train loss per epoch", t_loss)
        log_batch("val loss per epoch", v_loss)

        # ░░ save ckpt ░░
        improved = v_loss < best_val_loss if not np.isnan(v_loss) else False
        if improved:
            best_val_loss = v_loss
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "wandb_id": wandb_id,
        }
        torch.save(ckpt, latest_ckpt)
        if improved:
            torch.save(ckpt, os.path.join(config["save_dir"], f"best_model_{args.name}.pth"))

        # ░░ log & save audio every eval_interval ░░
        # ░░ log & save audio every eval_interval ░░
        if epoch % config["eval_interval"] == 0:
            with torch.no_grad():
                # parameter‐loss mode: only log synthetic audio
                if config["parameter_loss"]:
                    # generate one synthetic batch
                    audio, pitch, loud, _, _ = random_param_batch(
                        model.decoder,
                        config["batch_size"],
                        generator=train_gen,
                    )

                    audio = audio.to(dtype=torch.float32, device=device)
                    pitch = pitch.to(dtype=torch.float32, device=device)
                    loud = loud.to(dtype=torch.float32, device=device)

                    # get model reconstruction
                    rec = model(
                        pitch=pitch,
                        loudness=loud,
                        audio=audio,
                        audio_sr=config["sample_rate"],
                    )
                    # pick 5 examples
                    rand_idx = np.random.choice(audio.size(0), 5, replace=False)
                    for k, idx in enumerate(rand_idx):
                        sample = torch.cat([audio[idx], rec[idx]]).cpu().numpy()
                        # save locally if desired
                        sf.write(
                            os.path.join(
                                config["save_dir"],
                                f"param_sample_e{epoch}_{k}.wav"
                            ),
                            sample,
                            config["sample_rate"]
                        )
                        # log only these synthetic samples
                        wandb.log({
                            f"audio_param_compare_{k}": wandb.Audio(
                                sample,
                                sample_rate=config["sample_rate"],
                                caption=(
                                    f"epoch {epoch} | param sample {k} | "
                                    "original + recon"
                                )
                            )
                        })
                # normal mode: log Nsynth val‐set audio
                else:
                    a, p, l = next(iter(val_loader))
                    a, p, l = a.to(device), p.to(device), l.to(device)
                    rec = model(
                        pitch=p,
                        loudness=l,
                        audio=a,
                        audio_sr=config["sample_rate"],
                    )
                    rand_idx = np.random.choice(a.size(0), 5, replace=False)
                    for k, idx in enumerate(rand_idx):
                        sample = torch.cat([a[idx], rec[idx]]).cpu().numpy()
                        sf.write(
                            os.path.join(
                                config["save_dir"],
                                f"sample_e{epoch}_{k}.wav"
                            ),
                            sample,
                            config["sample_rate"]
                        )
                        wandb.log({
                            f"audio_compare_{k}": wandb.Audio(
                                sample,
                                sample_rate=config["sample_rate"],
                                caption=(
                                    f"epoch {epoch} | sample {k} | "
                                    "original + recon"
                                )
                            )
                        })

        print(f"[E{epoch}] train={t_loss:.4f} val={v_loss:.4f} best={best_val_loss:.4f}")

    wandb.finish()

if __name__ == '__main__':
    mp.freeze_support()
    main()