# Project: Train a DINOv2-based IP-Adapter for Stable Diffusion 1.5

(Feed this file to Claude Code: "Read BRIEF.md and follow it.")

## Goal
Adapt the official IP-Adapter training code to use DINOv2 as the image
encoder instead of CLIP, then run a small proving run to confirm the
pipeline works end-to-end before any full-scale training.

## Background (so you understand the design)
IP-Adapter conditions a frozen SD 1.5 UNet on a reference image. Only two
small things train: a projection network (resampler) and the per-layer
image cross-attention weights (W'_k, W'_v). Everything large — the SD UNet,
VAE, text encoder, and the image encoder — is FROZEN and downloaded
pretrained. We are NOT training a diffusion model from scratch.

I want to swap the frozen CLIP image encoder for frozen DINOv2, because
DINOv2's dense patch features capture finer visual detail. Known tradeoff:
DINOv2 is not text-aligned, so the text prompt may fight the image
conditioning more — we keep captions + CFG dropout to mitigate this.

## Step 1 — Set up
- Clone https://github.com/tencent-ailab/IP-Adapter
- READ `tutorial_train_plus.py` fully before editing. This is the
  resampler/"Plus" version — the right base for DINOv2 patch features.
  (Do NOT start from `tutorial_train.py`, the 4-token linear version.)
- Set up a Python env with: torch, diffusers, transformers, accelerate.
- SD 1.5 and DINOv2 weights auto-download from Hugging Face on first run.

## Step 2 — The five edits (the whole substance of the swap)
1. Replace the CLIP image encoder with `facebook/dinov2-large`, frozen.
2. Set the resampler's input/embedding dim to DINOv2's hidden size (1024
   for dinov2-large). A shape mismatch at the first forward pass is almost
   always here.
3. Feed DINOv2's PATCH tokens (optionally + CLS) into the resampler.
4. Switch image preprocessing to DINOv2's ImageNet normalization
   (mean/std), NOT CLIP's. THIS IS THE EASIEST THING TO GET SILENTLY
   WRONG — double-check it.
5. Keep image–text pairs in the data pipeline so the text branch and the
   ~5% classifier-free-guidance dropout still train.
Everything else (decoupled attention harness, loss = MSE on predicted
noise, optimizer) stays as-is.

## Step 3 — Dataset
- Format: a folder of images + a JSON listing each image's path + caption.
- For the proving run, a few thousand images is plenty. Use a small slice
  of an existing captioned set (e.g. a COCO or LAION subset), OR auto-caption
  my own images with BLIP. Help me wire whichever I pick into the script's
  expected JSON format.

## Step 4 — Proving run (do this BEFORE any long run)
Get training running end-to-end on the small dataset for a few thousand
steps. Success = data loads, dimensions line up, loss trends down, a
checkpoint saves. The goal is correct plumbing, not a good model.

## Step 5 — Verify conditioning works
Write a short inference script that loads the checkpoint and generates with
the IP-Adapter image scale (lambda) swept from 0 to 1. At 0 it should be
pure text-to-image; as lambda rises the reference image should visibly take
over. If nothing changes across the sweep, the conditioning isn't wired
right — debug before scaling up.

## My environment / constraints
- This runs on the KU Leuven HPC (Genius cluster) via the OnDemand portal.
- GPU: [FILL IN — e.g. the GPU on this interactive node, or the partition
  I'll submit to]
- IMPORTANT: on HPC, real training should run as a SLURM batch job, not in
  this login/interactive terminal. Help me write a SLURM submission script
  (sbatch) for the full run. The small proving run can run interactively if
  this node has a GPU.
- I'll watch VRAM myself. If we hit out-of-memory, drop batch size or
  enable gradient checkpointing — I'll tell you when it happens.
- This is my first time training a model like this, so explain
  non-obvious choices briefly as you go.

## Repo state
- I have already created/cloned this repo folder (IP-ADAPTER-TRAINING).
- Add a Python/ML `.gitignore` BEFORE the first commit: ignore checkpoints
  (*.bin, *.safetensors, *.pt, *.ckpt), the dataset images, the virtual
  environment, and __pycache__. Do not let checkpoints get committed.

## Working style
- Make the smallest changes that achieve each step; don't refactor the repo.
- After Step 1, summarize what the training loop does and where my five
  edits land, and wait for my OK before editing.
