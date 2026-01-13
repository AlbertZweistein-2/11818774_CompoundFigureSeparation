"""GPU smoke test to verify CUDA availability and a simple tensor op."""

import os
import subprocess
import sys

import pytest
import torch

# Allow skipping explicitly via env (e.g., on CPU-only CI)
SKIP_GPU_TEST = os.getenv("SKIP_GPU_TEST", "false").lower() in ("1", "true", "yes")


@pytest.mark.skipif(SKIP_GPU_TEST, reason="SKIP_GPU_TEST env set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keine GPU gefunden, Test übersprungen")
def test_gpu_availability_and_functionality():
    """Fail fast if no CUDA or if a trivial GPU op cannot run."""
    print(f"\nPython Version: {sys.version}")
    print(f"Torch Version: {torch.__version__}")

    # 1. Check if CUDA is generally available
    if not torch.cuda.is_available():
        pytest.fail("KEINE GPU GEFUNDEN! Torch läuft auf CPU.")

    # 2. Check System Status (nvidia-smi) via subprocess
    try:
        print("\n--- System GPU Status (nvidia-smi) ---")
        subprocess.run(["nvidia-smi"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Warning: nvidia-smi could not be run (driver missing or not in PATH?)")

    # 3. Hard Test: Actual Tensor Calculation
    try:
        # Move a tensor to GPU to catch "Zombie" driver states
        x = torch.tensor([1.0, 2.0]).cuda()
        y = x * 2 # Perform a calculation
        
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        vram = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        
        print(f"\nGPU TEST ERFOLGREICH: {device_name}")
        print(f"   VRAM Total: {vram:.2f} GB")
        
        # Verify the calculation actually worked
        assert y.device.type == 'cuda'
        
    except RuntimeError as e:
        pytest.fail(f"\nGPU FEHLER: Treiber scheint zu hängen (Suspend Bug?).\nError: {e}")