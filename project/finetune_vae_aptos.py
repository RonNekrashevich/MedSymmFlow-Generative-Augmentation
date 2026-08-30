"""Fine-tune an SD-VAE on fundus photographs (APTOS 2019) for LatMSF.

The generic photo VAE is the fidelity ceiling of every latent generator: if its
decoder smooths away fundus microstructure (microaneurysms, exudates), no flow
improvement can recover it. This adapts the full autoencoder to the fundus
domain with reconstruction + small KL (the LDM recipe), then saves a diffusers
directory usable directly as --gen-vae-id / --vae-id everywhere in the harness.

    python project/finetune_vae_aptos.py --images <dir with APTOS jpg/png> \
        --out /storage/medsymm/weights/vae_aptos --epochs 8

The saved config keeps SD-VAE's scaling_factor, which SymmFMClass reads at load.
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL

EXTS = {".png", ".jpg", ".jpeg"}


class ImageDirDataset(Dataset):
    def __init__(self, root, tf):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)
        assert self.paths, f"no images under {root}"
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune SD-VAE on a fundus image dir")
    ap.add_argument("--images", required=True, help="dir of native-resolution fundus images")
    ap.add_argument("--out", required=True, help="save_pretrained output dir")
    ap.add_argument("--base", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-weight", type=float, default=1e-6)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize(args.size),
        transforms.RandomResizedCrop(args.size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = ImageDirDataset(args.images, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"{len(ds)} images, {len(loader)} steps/epoch, size {args.size}")

    vae = AutoencoderKL.from_pretrained(args.base).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = torch.stack([ds[i] for i in range(min(8, len(ds)))]).to(device)

    def recon_of(x):
        posterior = vae.encode(x).latent_dist
        z = posterior.sample()
        return vae.decode(z).sample, posterior

    # Baseline reconstruction error of the UNTUNED VAE on this domain (report stat).
    vae.eval()
    with torch.no_grad():
        r0, _ = recon_of(fixed)
        base_mse = torch.nn.functional.mse_loss(r0, fixed).item()
    print(f"baseline (untuned) recon MSE on fundus: {base_mse:.5f}")
    save_image(torch.cat([fixed, r0]) * 0.5 + 0.5, out / "recon_epoch0.png", nrow=8)

    start = time.time()
    for epoch in range(args.epochs):
        vae.train()
        tot, n = 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                recon, posterior = recon_of(x)
                mse = torch.nn.functional.mse_loss(recon, x)
                kl = posterior.kl().mean()
                loss = mse + args.kl_weight * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += mse.item() * x.size(0)
            n += x.size(0)
        vae.eval()
        with torch.no_grad():
            r, _ = recon_of(fixed)
            fixed_mse = torch.nn.functional.mse_loss(r, fixed).item()
        print(f"epoch {epoch + 1}/{args.epochs}: train recon MSE {tot / n:.5f}, "
              f"fixed-batch MSE {fixed_mse:.5f} (baseline {base_mse:.5f})")
        save_image(torch.cat([fixed, r]) * 0.5 + 0.5, out / f"recon_epoch{epoch + 1}.png", nrow=8)
        vae.save_pretrained(out)   # overwrite each epoch: crash-safe, last = best-effort

    print(f"done in {(time.time() - start) / 60:.1f} min -> {out}")
    print(f"use with: --gen-vae-id {out}")


if __name__ == "__main__":
    main()
