# SAIN: Sketch-Aware Interpolation Network

This repository contains the official implementation for the paper **"Bridging the Gap: Sketch-Aware Interpolation Network for High-Quality Animation Sketch Inbetweening"**.

SAIN is a deep learning method designed to generate high-quality intermediate frames for hand-drawn sketch animations, addressing challenges like sparse strokes and exaggerated motions.

📄 **[Read the Paper (arXiv)](https://arxiv.org/abs/2308.13273)**

---

## 🚀 Quick Inference

We provide a robust standalone script (`inference.py`) to interpolate between two sketch frames. This script automatically handles:
* **Memory Management:** Automatic downscaling for high-resolution images to prevent CUDA OOM errors.
* **Aspect Ratio Preservation:** Uses smart padding to fit network requirements without distorting the image.
* **Dummy Inputs:** Automatically generates the required auxiliary inputs (points/region flow) for raw inference.

### Usage

**Basic Interpolation:**
```bash
python inference.py --img1 frame1.png --img2 frame2.png --output result.png
```

**For Large Images (Prevent OOM):**
Use the `--max_size` argument to limit the internal resolution (e.g., to 960px). The output will be upscaled back to the original resolution automatically.

```bash
python inference.py --img1 huge_frame1.png --img2 huge_frame2.png --output result.png --max_size 960
```

### Inference Arguments
* `--img1`: Path to the start frame (Keyframe 1).
* `--img2`: Path to the end frame (Keyframe 2).
* `--output`: Path to save the interpolated result.
* `--checkpoint`: Path to the `.pth` model (default: `ckp/checkpoints/model_best.pth`).
* `--max_size`: Maximum pixel size for the longest edge during processing (default: 960).

---

## 📥 Downloads & Setup

### 1. Pre-trained Models
Download the pre-trained model checkpoint to run inference or resume training.
* **[Download Model (Google Drive)](https://drive.google.com/file/d/1bPvGtm9Ty-ALrHc_NdzJQUes8elOBZfa/view?usp=sharing)**

Place the downloaded `model_best.pth` inside `ckp/checkpoints/`.

### 2. Dataset (STD-12K)
The Sketch Triplet Dataset (STD-12K) constructed for this research.
* **[Download Dataset Images (Google Drive)](https://drive.google.com/file/d/1vyu_ePFN9sFjqxc-sPdSWuSCLnWFVUT7/view?usp=sharing)**

### 3. Training Data (Correspondence)
Contains region & stroke level correspondence data required for training.
* **[Download Correspondence Data (Google Drive)](https://drive.google.com/file/d/1VMr2oPQCqUE579dnY4eFGGVAhrgjVR2V/view?usp=sharing)**

### Requirements
Ensure you have the following installed:
* Python 3.x
* PyTorch
* torchvision
* Pillow
* timm

---

## 🏋️ Training

To train the model from scratch using the STD-12K dataset:

```bash
python main.py --data_root /path/to/std12k_points --batch_size 4 --test_batch_size 4 --loss 0.7*L1+0.3*LPIPS
```

---

## 📚 Reference & Citation

If you find this code or dataset useful for your research, please cite our paper:

```bibtex
@inproceedings{shen2023sain,
    title={Bridging the Gap: Sketch-Aware Interpolation Network for High-Quality Animation Sketch Inbetweening},
    author={Jiaming Shen, Kun Hu, Wei Bao, Chang Wen Chen, and Zhiyong Wang},
    Booktitle = {Proc. of ACM International Conference on Multimedia (MM’24)},
    year={2024}
}
```

### Acknowledgements
We benefited from these excellent video interpolation resources:
* [AnimeInterp](https://github.com/lisiyao21/AnimeInterp.git)
* [VFI-Transformer](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer.git)
* [EISAI](https://github.com/ShuhongChen/eisai-anime-interpolator.git)