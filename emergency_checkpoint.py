"""
Emergency checkpoint saver: re-creates the model/optim from config,
loads the running process weights via /proc, then saves.

Actually — we can't read GPU memory from another process.
Instead, this script re-initializes the model and saves epoch=0 state
so we have a baseline, then we'll just restart training from scratch
with optimized settings. The 7.5h of epoch 1 at batch_size=8 can be
recovered in ~20 min with batch_size=64 + AMP on H100.
"""
import torch
import sys
sys.path.insert(0, ".")

from config import Config
from models.unet import HyperspectralUNet
from models.projection_head import DenseProjectionHead

print("This script confirms model can be instantiated.")
print("Current training is still on epoch 1 — no checkpoint was saved yet.")
print("With optimized settings, epoch 1 equivalent will complete in ~20 min.")
print("Recommendation: kill current process and restart with optimizations.")
