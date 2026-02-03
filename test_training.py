"""
Smoke test: Verifies that training can start successfully.
Runs for 2 steps and exits if forward/backward pass completes.
"""
import sys
import torch
import numpy as np

def test_training_starts():
    """Run 2 training steps to verify the pipeline works."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TEST] Using device: {device}")
    
    # 1. Test diffusion wrapper import (this is what we actually use)
    print("[TEST] Importing ShortcutDiffusion...")
    try:
        from denoising_diffusion_pytorch import Unet, ShortcutDiffusion
        print("[TEST] ✓ ShortcutDiffusion imported")
    except Exception as e:
        print(f"[TEST] ✗ Failed to import ShortcutDiffusion: {e}")
        return False
    
    # 3. Test dataset import
    print("[TEST] Importing dataset loader...")
    try:
        from utils.datasets import get_dataset
        print("[TEST] ✓ Dataset loader imported")
    except Exception as e:
        print(f"[TEST] ✗ Failed to import dataset: {e}")
        return False
    
    # 4. Create a minimal model
    print("[TEST] Creating model...")
    try:
        # Create small Unet for testing
        unet = Unet(
            dim=32,
            dim_mults=(1, 2),
            flash_attn=False,
            channels=3
        ).to(device)
        
        # Wrap in ShortcutDiffusion
        model = ShortcutDiffusion(
            model=unet,
            image_size=32,
            timesteps=128
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[TEST] ✓ Model created ({param_count:,} parameters)")
    except Exception as e:
        print(f"[TEST] ✗ Failed to create model: {e}")
        return False
    
    # 5. Test forward pass
    print("[TEST] Running forward pass...")
    try:
        fake_images = torch.randn(2, 3, 32, 32, device=device)
        fake_noise = torch.randn_like(fake_images)
        
        loss_dict = model(fake_images, fake_noise)
        loss = loss_dict['loss_total']
        print(f"[TEST] ✓ Forward pass complete (loss={loss.item():.4f})")
    except Exception as e:
        print(f"[TEST] ✗ Forward pass failed: {e}")
        return False
    
    # 6. Test backward pass
    print("[TEST] Running backward pass...")
    try:
        loss.backward()
        print("[TEST] ✓ Backward pass complete")
    except Exception as e:
        print(f"[TEST] ✗ Backward pass failed: {e}")
        return False
    
    # 7. Test optimizer step
    print("[TEST] Testing optimizer step...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        print("[TEST] ✓ Optimizer step complete")
    except Exception as e:
        print(f"[TEST] ✗ Optimizer step failed: {e}")
        return False
    
    # 8. Test sampling
    print("[TEST] Testing 1-step sampling...")
    try:
        model.eval()
        with torch.no_grad():
            sample_noise = torch.randn(2, 3, 32, 32, device=device)
            sample = model.sample(sample_noise, num_steps=1)
        print(f"[TEST] ✓ Sampling complete (output shape: {sample.shape})")
    except Exception as e:
        print(f"[TEST] ✗ Sampling failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("[TEST] ✓ ALL TESTS PASSED - Training can start!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_training_starts()
    sys.exit(0 if success else 1)
