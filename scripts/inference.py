#!/usr/bin/env python3
"""
Inference script for NAFNet image restoration.
Supports single image, batch processing, and tiled inference for large images.
"""

import os
import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from nafnet import nafnet_width32, nafnet_width64, nafnet_deblur


def load_model(model_type, weights_path, device):
    """Load model with weights"""
    print(f"Loading {model_type} model...")

    if model_type == "nafnet32":
        model = nafnet_width32()
    elif model_type == "nafnet64":
        model = nafnet_width64()
    else:
        model = nafnet_deblur()

    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def process_image(img, model, device, tile_size=None, tile_overlap=32):
    """
    Process a single image through the model.
    Uses tiled inference for large images to save memory.
    """
    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    h, w = img.shape[:2]

    with torch.no_grad():
        if tile_size is None or (h <= tile_size and w <= tile_size):
            # Direct inference
            output = model(img_tensor)
        else:
            # Tiled inference for large images
            output = tiled_inference(img_tensor, model, tile_size, tile_overlap)

    # Convert back to numpy
    output = output.squeeze(0).cpu().numpy()
    output = np.clip(output.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    return output


def tiled_inference(img_tensor, model, tile_size, overlap):
    """
    Process large images in tiles to save memory.
    """
    b, c, h, w = img_tensor.shape
    stride = tile_size - overlap

    # Calculate output size
    output = torch.zeros_like(img_tensor)
    count = torch.zeros_like(img_tensor)

    # Create weight map for blending (higher weight in center)
    weight = torch.ones(1, 1, tile_size, tile_size, device=img_tensor.device)

    # Cosine blending weights
    for i in range(overlap):
        weight[:, :, i, :] *= i / overlap
        weight[:, :, tile_size - 1 - i, :] *= i / overlap
        weight[:, :, :, i] *= i / overlap
        weight[:, :, :, tile_size - 1 - i] *= i / overlap

    # Process tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            # Ensure we don't go negative
            if y_start < 0:
                y_start = 0
                y_end = min(tile_size, h)
            if x_start < 0:
                x_start = 0
                x_end = min(tile_size, w)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]

            # Pad if necessary
            pad_h = tile_size - tile.shape[2]
            pad_w = tile_size - tile.shape[3]
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            # Process tile
            tile_output = model(tile)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                tile_output = tile_output[:, :, :tile_size-pad_h or None, :tile_size-pad_w or None]

            # Get actual tile size
            th, tw = tile_output.shape[2], tile_output.shape[3]
            tile_weight = weight[:, :, :th, :tw]

            # Add to output with weighting
            output[:, :, y_start:y_start+th, x_start:x_start+tw] += tile_output * tile_weight
            count[:, :, y_start:y_start+th, x_start:x_start+tw] += tile_weight

    # Normalize
    output = output / (count + 1e-8)

    return output


def process_single(input_path, output_path, model, device, tile_size):
    """Process a single image"""
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Could not read {input_path}")
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    result = process_image(img, model, device, tile_size)
    elapsed = time.time() - start_time

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)

    print(f"Processed {input_path.name} in {elapsed:.2f}s -> {output_path}")
    return True


def process_batch(input_dir, output_dir, model, device, tile_size):
    """Process a directory of images"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images = []
    for ext in extensions:
        images.extend(input_path.glob(ext))
        images.extend(input_path.glob(ext.upper()))

    print(f"Found {len(images)} images to process")

    for img_path in tqdm(images, desc="Processing"):
        out_path = output_path / f"{img_path.stem}_restored{img_path.suffix}"
        process_single(img_path, out_path, model, device, tile_size)


def create_comparison(input_path, output_path, model, device, tile_size, comparison_path):
    """Create side-by-side comparison image"""
    img = cv2.imread(str(input_path))
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = process_image(img_rgb, model, device, tile_size)

    # Create comparison (input | output)
    comparison = np.hstack([img_rgb, result])
    comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

    # Add labels
    h = comparison.shape[0]
    cv2.putText(comparison, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Output", (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(str(comparison_path), comparison)
    print(f"Comparison saved to {comparison_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="NAFNet inference for image restoration")

    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path")
    parser.add_argument("--model", type=str, default="nafnet32",
                        choices=["nafnet32", "nafnet64", "nafnet_deblur"],
                        help="Model architecture")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size for large images (default: auto based on VRAM)")
    parser.add_argument("--comparison", action="store_true",
                        help="Create side-by-side comparison")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

        # Auto tile size based on VRAM
        if args.tile_size is None:
            if vram < 8:
                args.tile_size = 256
            elif vram < 16:
                args.tile_size = 512
            elif vram < 24:
                args.tile_size = 768
            else:
                args.tile_size = None  # No tiling needed

    # Load model
    model = load_model(args.model, args.weights, device)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image
        if args.comparison:
            comparison_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
            create_comparison(input_path, output_path, model, device, args.tile_size, comparison_path)
            # Also save the restored image
            process_single(input_path, output_path, model, device, args.tile_size)
        else:
            process_single(input_path, output_path, model, device, args.tile_size)
    else:
        # Directory
        process_batch(input_path, output_path, model, device, args.tile_size)

    print("Done!")


if __name__ == "__main__":
    main()
