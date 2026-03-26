#!/usr/bin/env python3
"""
Dataset preparation script for image restoration training.
Downloads GoPro + RealBlur datasets and creates realistic degradation pairs.
"""

import os
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gdown
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import albumentations as A


# Dataset download URLs
DATASETS = {
    "gopro": {
        "url": "https://drive.google.com/uc?id=1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2",
        "filename": "GOPRO_Large.zip"
    },
    "realblur_j": {
        "url": "https://drive.google.com/uc?id=1Ycmaci-G5S4PR4ksUbo2KnNCeFPYImNO",
        "filename": "RealBlur_J.zip"
    },
    "realblur_r": {
        "url": "https://drive.google.com/uc?id=1qYiYLrZSm_EJDGdcvDepoE8GpbXiDaxG",
        "filename": "RealBlur_R.zip"
    }
}


class RealisticDegradation:
    """
    Creates realistic image degradations similar to Real-ESRGAN approach.
    Combines: blur + noise + compression + resize artifacts
    """

    def __init__(self, severity="medium"):
        self.severity = severity
        self.severity_params = {
            "light": {"blur": (0.5, 2.0), "noise": (5, 15), "jpeg": (60, 90)},
            "medium": {"blur": (1.0, 4.0), "noise": (10, 30), "jpeg": (30, 70)},
            "heavy": {"blur": (2.0, 7.0), "noise": (20, 50), "jpeg": (10, 50)}
        }
        self.params = self.severity_params[severity]

    def apply_gaussian_blur(self, img):
        sigma = random.uniform(*self.params["blur"])
        kernel_size = int(sigma * 4) | 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, 21))
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    def apply_motion_blur(self, img):
        size = random.randint(5, 20)
        angle = random.randint(0, 360)

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0
        kernel = kernel / size

        # Rotate kernel
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel = kernel / kernel.sum()

        return cv2.filter2D(img, -1, kernel)

    def apply_defocus_blur(self, img):
        """Disk/circle blur simulating defocus"""
        radius = random.randint(2, 8)
        kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel = kernel / kernel.sum()
        return cv2.filter2D(img, -1, kernel)

    def apply_noise(self, img):
        noise_level = random.uniform(*self.params["noise"])
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_jpeg_compression(self, img):
        quality = random.randint(*self.params["jpeg"])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def apply_resize_artifact(self, img):
        """Downscale then upscale to create resize artifacts"""
        h, w = img.shape[:2]
        scale = random.uniform(0.3, 0.7)

        # Random interpolation for down
        down_interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=down_interp)

        # Random interpolation for up
        up_interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
        return cv2.resize(small, (w, h), interpolation=up_interp)

    def degrade(self, img):
        """Apply random combination of degradations"""
        degraded = img.copy()

        # Randomly choose blur type
        blur_type = random.choice(["gaussian", "motion", "defocus", "none"])
        if blur_type == "gaussian":
            degraded = self.apply_gaussian_blur(degraded)
        elif blur_type == "motion":
            degraded = self.apply_motion_blur(degraded)
        elif blur_type == "defocus":
            degraded = self.apply_defocus_blur(degraded)

        # Additional degradations with probability
        if random.random() < 0.7:
            degraded = self.apply_noise(degraded)

        if random.random() < 0.5:
            degraded = self.apply_resize_artifact(degraded)

        if random.random() < 0.8:
            degraded = self.apply_jpeg_compression(degraded)

        return degraded


def download_dataset(dataset_name, output_dir):
    """Download and extract a dataset"""
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    info = DATASETS[dataset_name]
    output_path = Path(output_dir) / info["filename"]
    extract_path = Path(output_dir) / dataset_name

    if extract_path.exists():
        print(f"{dataset_name} already exists, skipping download")
        return extract_path

    print(f"Downloading {dataset_name}...")
    gdown.download(info["url"], str(output_path), quiet=False)

    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Clean up zip
    output_path.unlink()

    return extract_path


def process_gopro(gopro_path, output_dir):
    """Process GoPro dataset into train/val splits"""
    gopro_path = Path(gopro_path)
    output_path = Path(output_dir)

    for split in ["train", "test"]:
        split_path = gopro_path / split
        if not split_path.exists():
            # Try alternative structure
            split_path = gopro_path / "GOPRO_Large" / split

        if not split_path.exists():
            print(f"Could not find {split} split in GoPro dataset")
            continue

        out_split = "train" if split == "train" else "val"
        blur_out = output_path / out_split / "blur"
        sharp_out = output_path / out_split / "sharp"
        blur_out.mkdir(parents=True, exist_ok=True)
        sharp_out.mkdir(parents=True, exist_ok=True)

        # Find all sequences
        for seq_dir in tqdm(list(split_path.iterdir()), desc=f"Processing GoPro {split}"):
            if not seq_dir.is_dir():
                continue

            blur_dir = seq_dir / "blur"
            sharp_dir = seq_dir / "sharp"

            if not blur_dir.exists() or not sharp_dir.exists():
                continue

            for blur_img in blur_dir.glob("*.png"):
                sharp_img = sharp_dir / blur_img.name
                if sharp_img.exists():
                    # Copy with unique name
                    new_name = f"{seq_dir.name}_{blur_img.name}"
                    shutil.copy(blur_img, blur_out / new_name)
                    shutil.copy(sharp_img, sharp_out / new_name)


def process_realblur(realblur_path, output_dir, subset="J"):
    """Process RealBlur dataset"""
    realblur_path = Path(realblur_path)
    output_path = Path(output_dir)

    for split in ["train", "test"]:
        split_path = realblur_path / split
        if not split_path.exists():
            split_path = realblur_path / f"RealBlur_{subset}" / split

        if not split_path.exists():
            print(f"Could not find {split} in RealBlur")
            continue

        out_split = "train" if split == "train" else "val"
        blur_out = output_path / out_split / "blur"
        sharp_out = output_path / out_split / "sharp"
        blur_out.mkdir(parents=True, exist_ok=True)
        sharp_out.mkdir(parents=True, exist_ok=True)

        # RealBlur structure: split/scene/blur/ and split/scene/gt/
        for scene in tqdm(list(split_path.iterdir()), desc=f"Processing RealBlur-{subset} {split}"):
            if not scene.is_dir():
                continue

            blur_dir = scene / "blur"
            gt_dir = scene / "gt"

            if not blur_dir.exists() or not gt_dir.exists():
                continue

            for blur_img in blur_dir.glob("*.png"):
                gt_img = gt_dir / blur_img.name
                if not gt_img.exists():
                    gt_img = gt_dir / blur_img.name.replace("blur", "gt")

                if gt_img.exists():
                    new_name = f"realblur_{subset}_{scene.name}_{blur_img.name}"
                    shutil.copy(blur_img, blur_out / new_name)
                    shutil.copy(gt_img, sharp_out / new_name)


def create_synthetic_pairs(source_dir, output_dir, num_per_image=3, severity="medium"):
    """Create synthetic degraded pairs from sharp images"""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    blur_out = output_path / "train" / "blur"
    sharp_out = output_path / "train" / "sharp"
    blur_out.mkdir(parents=True, exist_ok=True)
    sharp_out.mkdir(parents=True, exist_ok=True)

    degrader = RealisticDegradation(severity)

    # Find all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images = []
    for ext in extensions:
        images.extend(source_path.rglob(ext))

    print(f"Found {len(images)} images to process")

    for img_path in tqdm(images, desc="Creating synthetic pairs"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Skip if too small
        h, w = img.shape[:2]
        if h < 256 or w < 256:
            continue

        for i in range(num_per_image):
            degraded = degrader.degrade(img)

            name = f"synthetic_{img_path.stem}_{i}.png"
            cv2.imwrite(str(sharp_out / name), img)
            cv2.imwrite(str(blur_out / name), degraded)


def create_patches(input_dir, output_dir, patch_size=256, stride=128):
    """Create training patches from full images"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for split in ["train", "val"]:
        blur_in = input_path / split / "blur"
        sharp_in = input_path / split / "sharp"

        if not blur_in.exists():
            continue

        blur_out = output_path / split / "blur"
        sharp_out = output_path / split / "sharp"
        blur_out.mkdir(parents=True, exist_ok=True)
        sharp_out.mkdir(parents=True, exist_ok=True)

        blur_images = list(blur_in.glob("*.png")) + list(blur_in.glob("*.jpg"))

        for blur_img_path in tqdm(blur_images, desc=f"Creating {split} patches"):
            sharp_img_path = sharp_in / blur_img_path.name
            if not sharp_img_path.exists():
                continue

            blur_img = cv2.imread(str(blur_img_path))
            sharp_img = cv2.imread(str(sharp_img_path))

            if blur_img is None or sharp_img is None:
                continue

            h, w = blur_img.shape[:2]

            patch_idx = 0
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    blur_patch = blur_img[y:y+patch_size, x:x+patch_size]
                    sharp_patch = sharp_img[y:y+patch_size, x:x+patch_size]

                    name = f"{blur_img_path.stem}_patch{patch_idx}.png"
                    cv2.imwrite(str(blur_out / name), blur_patch)
                    cv2.imwrite(str(sharp_out / name), sharp_patch)
                    patch_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for image restoration training")
    parser.add_argument("--output", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--download-gopro", action="store_true", help="Download GoPro dataset")
    parser.add_argument("--download-realblur", action="store_true", help="Download RealBlur dataset")
    parser.add_argument("--custom-images", type=str, help="Path to custom sharp images for synthetic pairs")
    parser.add_argument("--synthetic-severity", type=str, default="medium",
                        choices=["light", "medium", "heavy"], help="Degradation severity")
    parser.add_argument("--create-patches", action="store_true", help="Create training patches")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size")
    parser.add_argument("--patch-stride", type=int, default=128, help="Stride between patches")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    patches_dir = output_dir / "patches"

    # Download datasets
    if args.download_gopro:
        gopro_path = download_dataset("gopro", raw_dir)
        if gopro_path:
            process_gopro(raw_dir, processed_dir)

    if args.download_realblur:
        realblur_j = download_dataset("realblur_j", raw_dir)
        realblur_r = download_dataset("realblur_r", raw_dir)
        if realblur_j:
            process_realblur(raw_dir / "realblur_j", processed_dir, "J")
        if realblur_r:
            process_realblur(raw_dir / "realblur_r", processed_dir, "R")

    # Create synthetic pairs from custom images
    if args.custom_images:
        print(f"Creating synthetic pairs from {args.custom_images}")
        create_synthetic_pairs(args.custom_images, processed_dir,
                              severity=args.synthetic_severity)

    # Create patches for training
    if args.create_patches:
        print("Creating training patches...")
        create_patches(processed_dir, patches_dir,
                      args.patch_size, args.patch_stride)
        print(f"Patches saved to {patches_dir}")

    # Print statistics
    for split in ["train", "val"]:
        blur_dir = processed_dir / split / "blur"
        if blur_dir.exists():
            count = len(list(blur_dir.glob("*")))
            print(f"{split}: {count} image pairs")

    if args.create_patches:
        for split in ["train", "val"]:
            blur_dir = patches_dir / split / "blur"
            if blur_dir.exists():
                count = len(list(blur_dir.glob("*")))
                print(f"{split} patches: {count}")


if __name__ == "__main__":
    main()
