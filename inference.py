"""
SAIN Inference Script (Stand-alone)

This script performs frame interpolation between two input images using a pre-trained SAIN model.
It handles memory constraints automatically by downscaling large images and restoring them
to the original resolution and aspect ratio after inference.

Example Usage:
    python inference.py --img1 frame0.png --img2 frame1.png --output result.png --max_size 960
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import OrderedDict

# Add current directory to path to ensure module imports work correctly
sys.path.append(os.getcwd())

try:
    from model.SAIN import SAIN
except ImportError:
    print("Error: Could not import SAIN. Make sure you are running this script from the repository root.")
    sys.exit(1)

# --- CONFIGURATION CLASS ---
# Mimics the arguments object used during training to satisfy SAIN's constructor
class InferenceConfig:
    def __init__(self):
        # Hardware settings
        self.cuda = True
        self.num_gpu = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Paths (Empty strings to prevent the model from seeking training files)
        self.resume_flownet = ''
        self.resume = False

        # Model Configuration
        self.phase = 'test'
        self.dataset = 'std12k'
        self.joinType = 'concat'
        self.model = 'SAIN'

        # Network Structure
        self.nbr_frame = 2
        self.crop_size = 192
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_workers = 1

        # Dummy Training Parameters (Required by __init__ but unused in inference)
        self.loss = '0.7*L1+0.3*LPIPS'
        self.lr = 0.0002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.start_epoch = 0
        self.max_epoch = 50
        self.pretrained = None
        self.checkpoint_dir = 'ckp'
        self.result_dir = 'result'
        self.log_iter = 100

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert('RGB')
    return img

def save_image(tensor, path):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]

    # Post-process: Remove batch dim, clamp to [0,1], convert to PIL
    img = tensor.squeeze(0).cpu().clamp(0, 1)
    tf = transforms.ToPILImage()
    img_pil = tf(img)
    img_pil.save(path)
    print(f"✅ Output saved to: {path}")

def run_inference(args):
    print(f"🔧 Setting up SAIN model (Max resolution: {args.max_size}px)...")

    # 1. Initialize Model Configuration
    model_args = InferenceConfig()
    try:
        model = SAIN(model_args)
    except Exception as e:
        print(f"❌ Error initializing SAIN model: {e}")
        return

    # 2. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found at: {args.checkpoint}")
        return

    print(f"📂 Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=model_args.device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Clean up 'module.' prefix (caused by DataParallel training)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    # Load weights
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        print(f"⚠️  Strict load failed. Attempting flexible load (ignoring non-critical keys)...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(model_args.device)
    model.eval()

    # 3. Process Input Images (Downscale + Padding logic)
    print(f"🖼️  Processing input images...")
    img1_pil = load_image(args.img1)
    img2_pil = load_image(args.img2)

    orig_w, orig_h = img1_pil.size

    # Calculate Downscaling Factor (To prevent OOM on large images)
    scale_factor = 1.0
    if max(orig_w, orig_h) > args.max_size:
        scale_factor = args.max_size / max(orig_w, orig_h)
        new_w_scaled = int(orig_w * scale_factor)
        new_h_scaled = int(orig_h * scale_factor)
        print(f"📉 Downscaling input: {orig_w}x{orig_h} -> {new_w_scaled}x{new_h_scaled} (Factor: {scale_factor:.2f})")

        img1_pil = img1_pil.resize((new_w_scaled, new_h_scaled), Image.BICUBIC)
        img2_pil = img2_pil.resize((new_w_scaled, new_h_scaled), Image.BICUBIC)
    else:
        new_w_scaled, new_h_scaled = orig_w, orig_h

    # Convert to Tensor
    tf = transforms.ToTensor()
    I0 = tf(img1_pil).unsqueeze(0).to(model_args.device)
    I1 = tf(img2_pil).unsqueeze(0).to(model_args.device)

    # Calculate Padding (Must be multiple of 64 for U-Net architecture)
    align = 64
    pad_w = (align - (new_w_scaled % align)) % align
    pad_h = (align - (new_h_scaled % align)) % align

    # Apply Padding (Replicate mode to preserve aspect ratio logic)
    if pad_w > 0 or pad_h > 0:
        print(f"⬜ Applying padding: W+{pad_w}, H+{pad_h}")
        # Pad format: (left, right, top, bottom)
        I0 = F.pad(I0, (0, pad_w, 0, pad_h), mode='replicate')
        I1 = F.pad(I1, (0, pad_w, 0, pad_h), mode='replicate')

    # 4. Generate Dummy Inputs
    # SAIN requires 'points' and 'region_flow' as inputs.
    # For raw inference, we provide zero-tensors with correct shapes.
    B, _, H_pad, W_pad = I0.shape

    # Points: Expects 1-channel mask (Batch, 1, H, W)
    dummy_points = torch.zeros(B, 1, H_pad, W_pad).to(model_args.device)

    # Region Flow: Expects a list of 2 tensors (Forward/Backward), each with 2 channels
    flow_tensor = torch.zeros(B, 2, H_pad, W_pad).to(model_args.device)
    dummy_region_flow = [flow_tensor, flow_tensor]

    # 5. Inference
    print("🚀 Generating intermediate frame...")
    with torch.no_grad():
        output = model(I0, I1, dummy_points, dummy_region_flow)

        # Handle list/tuple output
        if isinstance(output, (list, tuple)):
            output = output[0]

    # 6. Post-processing (Crop Padding + Restore Resolution)
    print("✨ Restoring original resolution...")

    # a) Remove padding
    crop_w = W_pad - pad_w
    crop_h = H_pad - pad_h
    output = output[:, :, :crop_h, :crop_w]

    # b) Upscale to exact original dimensions
    output = F.interpolate(output, size=(orig_h, orig_w), mode='bicubic', align_corners=False)

    save_image(output, args.output)

if __name__ == "__main__":
    example_text = """
Example:
    python inference.py --img1 frame1.png --img2 frame2.png --output result.png

    # For large images (prevents Out of Memory):
    python inference.py --img1 frame1.png --img2 frame2.png --output result.png --max_size 960
    """

    parser = argparse.ArgumentParser(
        description="SAIN Inference Script",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--img1', type=str, help='Path to the first keyframe (Start)')
    parser.add_argument('--img2', type=str, help='Path to the second keyframe (End)')
    parser.add_argument('--checkpoint', type=str, default='./ckp/checkpoints/model_best.pth', help='Path to the .pth model checkpoint')
    parser.add_argument('--output', type=str, default='result.png', help='Path for the output interpolated image')
    parser.add_argument('--max_size', type=int, default=960, help='Max resolution size to avoid OOM errors (default: 960)')

    # If run without arguments, print help and example, then exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if not args.img1 or not args.img2:
        print("❌ Error: You must provide --img1 and --img2 arguments.")
        sys.exit(1)

    run_inference(args)
