"""
Generate a promo video for Sharp Restore app using ComfyUI API.
"""

import json
import urllib.request
import urllib.parse
import time
import sys
import os

COMFYUI_URL = "http://127.0.0.1:8188"

# Workflow for app promo video
WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "dreamshaper_8.safetensors"
        }
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["1", 1],
            "text": "A sleek modern app interface for image restoration floating in 3D space, clean minimal dark design, glowing blue UI elements, smooth camera rotation around the interface, tech startup aesthetic, cinematic lighting with soft shadows, motion blur effect, 4k quality, professional product showcase, dark gradient background with subtle floating particles, Sharp Restore logo, before and after image comparison"
        }
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["1", 1],
            "text": "ugly, blurry, low quality, distorted, watermark, text overlay, bad anatomy, worst quality, jpeg artifacts, noise, grainy, oversaturated, amateur, unprofessional, static, no motion"
        }
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 16
        }
    },
    "5": {
        "class_type": "ADE_AnimateDiffLoaderWithContext",
        "inputs": {
            "model": ["1", 0],
            "model_name": "v3_sd15_mm.ckpt",
            "beta_schedule": "sqrt_linear (AnimateDiff)",
            "motion_scale": 1.0,
            "apply_v2_models_properly": False,
            "context_options": None
        }
    },
    "6": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["5", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 42,
            "steps": 25,
            "cfg": 7.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0
        }
    },
    "7": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": ["1", 2]
        }
    },
    "8": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["7", 0],
            "frame_rate": 8,
            "loop_count": 0,
            "filename_prefix": "sharp_restore_promo",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True
        }
    }
}


def check_server():
    """Check if ComfyUI server is running"""
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/system_stats")
        urllib.request.urlopen(req, timeout=5)
        return True
    except:
        return False


def queue_prompt(workflow):
    """Submit workflow to ComfyUI"""
    data = json.dumps({"prompt": workflow}).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def get_history(prompt_id):
    """Get generation history"""
    req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def main():
    print("=" * 50)
    print("  Sharp Restore - Promo Video Generator")
    print("=" * 50)
    print()

    # Wait for server
    print("Waiting for ComfyUI server...")
    attempts = 0
    while not check_server() and attempts < 60:
        time.sleep(2)
        attempts += 1
        print(f"  Attempt {attempts}/60...")

    if not check_server():
        print("ERROR: ComfyUI server not responding.")
        print("Please start ComfyUI manually:")
        print("  D:\\ComfyUI\\ComfyUI_windows_portable\\run_nvidia_gpu.bat")
        sys.exit(1)

    print("Server is ready!")
    print()

    # Queue the prompt
    print("Submitting video generation request...")
    result = queue_prompt(WORKFLOW)
    prompt_id = result.get("prompt_id")
    print(f"  Job ID: {prompt_id}")
    print()

    # Wait for completion
    print("Generating video (this may take a few minutes)...")
    while True:
        time.sleep(3)
        history = get_history(prompt_id)

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            if outputs:
                print()
                print("=" * 50)
                print("  Video generated successfully!")
                print("=" * 50)
                print()
                print("Output saved to:")
                print("  D:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI\\output\\")
                print()

                # Find video file
                for node_id, output in outputs.items():
                    if "gifs" in output:
                        for gif in output["gifs"]:
                            filename = gif.get("filename", "")
                            print(f"  Video: {filename}")
                break

        print(".", end="", flush=True)


if __name__ == "__main__":
    main()
