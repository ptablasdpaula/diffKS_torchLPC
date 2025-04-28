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

from paths import NSYNTH_PREPROCESSED_DIR

def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def parse_args():
    p = argparse.ArgumentParser()
    env = os.environ.get

    p.add_argument("--name", type=str, default=env("NAME", "exp"),
                        help="Unique experiment name (used for checkpoints & wandb run)")
    p.add_argument("--continue_from_checkpoint", action="store_true",
                        default=str2bool(env("CONTINUE_FROM_CHECKPOINT", "false")),
                        help="Resume training from latest checkpoint for this --name")

    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-4)))

    p.add_argument("--batch_size",  type=int, default=int(env("BATCH_SIZE", 16)))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 4)))

    p.add_argument("--hidden_size", type=int, default=int(env("HIDDEN_SIZE", 512)))

    p.add_argument("--l_order",     type=int, default=int(env("L_ORDER", 2)))
    p.add_argument("--l_n_frames",  type=int, default=int(env("L_N_FRAMES", 250)))
    p.add_argument("--exc_order",   type=int, default=int(env("EXC_ORDER", 10)))
    p.add_argument("--exc_n_frames",type=int, default=int(env("EXC_N_FRAMES", 100)))

    p.add_argument("--families",    type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources",     type=str, default=env("SOURCES", "acoustic"))

    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "linear"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

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
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": 200,
        "eval_interval": 1,
        "save_dir": f"runs/ks_nsynth/{args.name}",
        "families": [f.strip() for f in args.families.split(",")],
        "sources": [s.strip() for s in args.sources.split(",")],
        "num_workers": args.num_workers,
        "interpolation_type": args.interpolation_type,
        "pitch_mode": args.pitch_mode,
    }

    print("\n▶Running with config:")
    for k, v in vars(args).items():
        print(f"   {k:12}: {v}")

    device = get_device()
    print(f"Using device: {device}")

    # ─── WandB init ────────────────────────────────────────────────────────
    wandb_id = None
    latest_ckpt = os.path.join(config["save_dir"], f"latest_model_{args.name}.pth")
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        tmp_ckpt = torch.load(latest_ckpt, map_location="cpu")
        wandb_id = tmp_ckpt.get("wandb_id", None)
        print(f"[INFO] Found checkpoint – will resume run id {wandb_id}")

    wandb_run = wandb.init(project="diffks-autoencoder", name=args.name,
                           id=wandb_id, resume="allow", config=config)
    if wandb_id is None:
        wandb_id = wandb_run.id  # store for fresh runs

    # ─── RAM check ────────────────────────────────────────────── #
    process = psutil.Process(os.getpid())
    print(f"[INFO] Memory at start: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    # ─── Create save directories ───────────────────────────────── #
    full_save_path = os.path.abspath(config["save_dir"])
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Using save directory: {full_save_path}")

    # ─── Data ─────────────────────────────────────────────────────────────
    dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                            split="train",
                            pitch_mode=config["pitch_mode"],
                            families=config["families"],
                            sources=config["sources"], )

    val_dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                                split="valid",
                                pitch_mode=config["pitch_mode"],
                                families=config["families"],
                                sources=config["sources"], )

    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, pin_memory=True if device.type != "mps" else False, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            drop_last=True, pin_memory=True if device.type != "mps" else False, num_workers=config["num_workers"])

    # ─── Start Model, optimizer & Loss ────────────────────────── #
    model = AE_KarplusModel(batch_size=config["batch_size"], hidden_size=config["hidden_size"],
                            loop_order=config["loop_order"], loop_n_frames=config["loop_n_frames"],
                            exc_order=config["exc_order"], exc_n_frames=config["exc_n_frames"],
                            sample_rate=config["sample_rate"], interpolation_type=config["interpolation_type"],
                            z_encoder=MfccTimeDistributedRnnEncoder(),).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    mr_stft = MultiResolutionSTFTLoss(scale_invariance=True, perceptual_weighting=True,
                                      sample_rate=config["sample_rate"], device=device, )

    # ─── Resume from checkpoint if requested ──────────────────────────────
    start_epoch, best_val_loss = 0, float('inf')
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"[RESUME] Starting at epoch {start_epoch} (best so far {best_val_loss:.4f})")

    # ───────────────────────── training epochs ───────────────────────────
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        t_loss = 0
        for audio, pitch, loud in tqdm(train_loader, desc=f"[E{epoch:03d} train]"):
            audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
            recon = model(pitch=pitch, loudness=loud, audio=audio)
            loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # ░░ VALID ░░ (every eval_interval)
        if epoch % config["eval_interval"] == 0:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for audio, pitch, loud in val_loader:
                    audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                    recon = model(pitch=pitch, loudness=loud, audio=audio)
                    v_losses.append(mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)).item())
            v_loss = float(np.mean(v_losses))
        else:
            v_loss = np.nan

        # ░░ logging ░░
        wandb.log({"train_loss": t_loss, "val_loss": v_loss, "epoch": epoch})

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
        if epoch % config["eval_interval"] == 0:
            with torch.no_grad():
                a, p, l = next(iter(val_loader))
                a, p, l = a.to(device), p.to(device), l.to(device)
                rec = model(pitch=p, loudness=l, audio=a)
                # --- pick 5 unique indices from the batch (assumes batch_size ≥ 5) ---
                rand_idx = np.random.choice(a.size(0), 5, replace=False)

                for k, idx in enumerate(rand_idx):
                    sample = torch.cat([a[idx], rec[idx]]).cpu().numpy()
                    sf.write(
                        os.path.join(config["save_dir"], f"sample_e{epoch}_{k}.wav"),
                        sample, config["sample_rate"]
                    )

                    wandb.log({
                        f"audio_compare_{k}": wandb.Audio(
                            sample,
                            sample_rate=config["sample_rate"],
                            caption=f"epoch {epoch} | sample {k} | original + recon"
                        )
                    })

        print(f"[E{epoch}] train={t_loss:.4f} val={v_loss:.4f} best={best_val_loss:.4f}")

    wandb.finish()

if __name__ == '__main__':
    mp.freeze_support()
    main()