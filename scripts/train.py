#!/usr/bin/env python3
"""
Training script for NAFNet image restoration model.
Optimized for RunPod with mixed precision and efficient data loading.
"""

import os
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import cv2
from PIL import Image
from tqdm import tqdm
import lpips

from nafnet import nafnet_width32, nafnet_width64, nafnet_deblur


# ============== Dataset ==============

class ImageRestorationDataset(Dataset):
    """Dataset for paired blur/sharp images"""

    def __init__(self, blur_dir, sharp_dir, patch_size=256, augment=True):
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.patch_size = patch_size
        self.augment = augment

        # Find all image pairs
        self.blur_images = sorted(list(self.blur_dir.glob("*.png")) +
                                  list(self.blur_dir.glob("*.jpg")))

        # Filter to only include pairs where both exist
        self.pairs = []
        for blur_path in self.blur_images:
            sharp_path = self.sharp_dir / blur_path.name
            if sharp_path.exists():
                self.pairs.append((blur_path, sharp_path))

        print(f"Found {len(self.pairs)} image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        # Load images
        blur = cv2.imread(str(blur_path))
        sharp = cv2.imread(str(sharp_path))

        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        # Random crop
        h, w = blur.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            blur = blur[top:top+self.patch_size, left:left+self.patch_size]
            sharp = sharp[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if too small
            blur = cv2.resize(blur, (self.patch_size, self.patch_size))
            sharp = cv2.resize(sharp, (self.patch_size, self.patch_size))

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                blur = np.fliplr(blur).copy()
                sharp = np.fliplr(sharp).copy()

            # Random vertical flip
            if random.random() > 0.5:
                blur = np.flipud(blur).copy()
                sharp = np.flipud(sharp).copy()

            # Random 90 degree rotation
            k = random.randint(0, 3)
            if k > 0:
                blur = np.rot90(blur, k).copy()
                sharp = np.rot90(sharp, k).copy()

        # To tensor [0, 1]
        blur = torch.from_numpy(blur.transpose(2, 0, 1)).float() / 255.0
        sharp = torch.from_numpy(sharp.transpose(2, 0, 1)).float() / 255.0

        return blur, sharp


class ValidationDataset(Dataset):
    """Validation dataset without augmentation"""

    def __init__(self, blur_dir, sharp_dir, max_size=512):
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.max_size = max_size

        self.blur_images = sorted(list(self.blur_dir.glob("*.png")) +
                                  list(self.blur_dir.glob("*.jpg")))

        self.pairs = []
        for blur_path in self.blur_images:
            sharp_path = self.sharp_dir / blur_path.name
            if sharp_path.exists():
                self.pairs.append((blur_path, sharp_path))

    def __len__(self):
        return min(len(self.pairs), 100)  # Limit validation set

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        blur = cv2.imread(str(blur_path))
        sharp = cv2.imread(str(sharp_path))

        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        # Resize if too large
        h, w = blur.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            blur = cv2.resize(blur, (new_w, new_h))
            sharp = cv2.resize(sharp, (new_w, new_h))

        # Make divisible by 16
        h, w = blur.shape[:2]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        blur = blur[:new_h, :new_w]
        sharp = sharp[:new_h, :new_w]

        blur = torch.from_numpy(blur.transpose(2, 0, 1)).float() / 255.0
        sharp = torch.from_numpy(sharp.transpose(2, 0, 1)).float() / 255.0

        return blur, sharp


# ============== Losses ==============

class PSNRLoss(nn.Module):
    """PSNR Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        return -10 * torch.log10(mse + 1e-8)


class SSIMLoss(nn.Module):
    """SSIM Loss"""
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)

        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred*pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target*target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred*target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using LPIPS"""
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg', verbose=False)

    def forward(self, pred, target):
        # LPIPS expects [-1, 1] range
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1
        return self.lpips(pred_scaled, target_scaled).mean()


class CombinedLoss(nn.Module):
    """Combined loss: L1 + SSIM + Perceptual"""
    def __init__(self, l1_weight=1.0, ssim_weight=0.1, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss() if perceptual_weight > 0 else None

    def forward(self, pred, target):
        loss = self.l1_weight * self.l1(pred, target)
        loss += self.ssim_weight * self.ssim(pred, target)

        if self.perceptual is not None and self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual(pred, target)

        return loss


# ============== Metrics ==============

def calculate_psnr(pred, target):
    """Calculate PSNR"""
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1 / (mse + 1e-8))


def calculate_ssim(pred, target):
    """Calculate SSIM"""
    ssim_loss = SSIMLoss()
    return 1 - ssim_loss(pred, target)


# ============== Training ==============

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for blur, sharp in pbar:
        blur = blur.to(device)
        sharp = sharp.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(blur)
            loss = criterion(pred, sharp)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for blur, sharp in tqdm(dataloader, desc="Validation"):
            blur = blur.to(device)
            sharp = sharp.to(device)

            pred = model(blur)
            pred = torch.clamp(pred, 0, 1)

            total_psnr += calculate_psnr(pred, sharp).item()
            total_ssim += calculate_ssim(pred, sharp).item()

    return {
        "psnr": total_psnr / len(dataloader),
        "ssim": total_ssim / len(dataloader)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def main():
    parser = argparse.ArgumentParser(description="Train NAFNet for image restoration")

    # Data
    parser.add_argument("--train-blur", type=str, required=True, help="Training blur images directory")
    parser.add_argument("--train-sharp", type=str, required=True, help="Training sharp images directory")
    parser.add_argument("--val-blur", type=str, help="Validation blur images directory")
    parser.add_argument("--val-sharp", type=str, help="Validation sharp images directory")

    # Model
    parser.add_argument("--model", type=str, default="nafnet32", choices=["nafnet32", "nafnet64", "nafnet_deblur"],
                        help="Model architecture")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, help="Load pretrained weights")

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=256, help="Training patch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")

    # Loss weights
    parser.add_argument("--l1-weight", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument("--ssim-weight", type=float, default=0.1, help="SSIM loss weight")
    parser.add_argument("--perceptual-weight", type=float, default=0.05, help="Perceptual loss weight")

    # Other
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")

    # Model
    print(f"Creating model: {args.model}")
    if args.model == "nafnet32":
        model = nafnet_width32()
    elif args.model == "nafnet64":
        model = nafnet_width64()
    else:
        model = nafnet_deblur()

    model = model.to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params / 1e6:.2f}M")

    # Load pretrained weights
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # Dataset
    print("Loading datasets...")
    train_dataset = ImageRestorationDataset(
        args.train_blur, args.train_sharp,
        patch_size=args.patch_size, augment=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = None
    if args.val_blur and args.val_sharp:
        val_dataset = ValidationDataset(args.val_blur, args.val_sharp)
        val_loader = DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, num_workers=2
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.9)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    # Loss
    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight
    ).to(device)

    # Mixed precision
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        best_psnr = metrics.get("best_psnr", 0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch + 1)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Validate
        if val_loader:
            metrics = validate(model, val_loader, device)
            writer.add_scalar("Metrics/PSNR", metrics["psnr"], epoch)
            writer.add_scalar("Metrics/SSIM", metrics["ssim"], epoch)

            print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

            # Save best
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"best_psnr": best_psnr, **metrics},
                    checkpoint_dir / "best.pth"
                )
                print(f"  -> New best PSNR: {best_psnr:.2f}dB")
        else:
            print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}")

        # Update scheduler
        scheduler.step()

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"best_psnr": best_psnr},
                checkpoint_dir / f"epoch_{epoch + 1}.pth"
            )

    # Save final
    save_checkpoint(
        model, optimizer, scheduler, args.epochs,
        {"best_psnr": best_psnr},
        checkpoint_dir / "final.pth"
    )

    # Export model only (for inference)
    torch.save(model.state_dict(), output_dir / "model_weights.pth")

    print(f"\nTraining complete!")
    print(f"Best PSNR: {best_psnr:.2f}dB")
    print(f"Model saved to {output_dir}")

    writer.close()


if __name__ == "__main__":
    main()
