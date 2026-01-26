import torch
import numpy as np
import os
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def eval_model(args, model, model_teacher, step, valid_loader, device='cuda'):
    """
    Simplified evaluation: Generates samples with different step counts.
    """
    print(f"--- Evaluation at Step {step} ---")
    
    # 1. Setup
    model.eval()
    num_samples = 16
    img_size = args.model_config.image_size
    channels = args.model_config.image_channels
    
    # Get labels for conditional generation (0 to 15)
    # y = torch.arange(num_samples, device=device) % args.model_config.num_classes
    # Or just random labels
    y = torch.randint(0, args.model_config.num_classes, (num_samples,), device=device)
    
    # Start from noise
    x_start = torch.randn(num_samples, channels, img_size, img_size, device=device)
    
    # Steps to test: 1-step (Shortcut), 4-step, 32-step (ODE ref)
    steps_list = [1, 2, 4, 8, 32]
    
    generated_images = []
    titles = []

    with torch.no_grad():
        for steps in steps_list:
            print(f"Generating with {steps} steps...")
            
            # Euler Sampler
            dt = 1.0 / steps
            x = x_start.clone()
            
            for i in range(steps):
                t_val = i / steps
                t = torch.ones(num_samples, device=device) * t_val
                
                # For Shortcut models, dt_base corresponds to log2(steps) roughly
                # Or we can just pass the "Target" dt index.
                # If steps=1, dt=1.0 -> index 0. If steps=2, dt=0.5 -> index 1.
                # dt = 1/2^k. log2(1/dt) = k.
                dt_k = int(np.log2(steps)) # This is an approximation
                dt_base = torch.ones(num_samples, device=device, dtype=torch.long) * dt_k
                
                # Model Prediction
                # v = model(x, t, dt_base, y)
                # Using 0 CFG for basic check first
                v = model(x, t, dt_base, y, train=False)
                
                # Euler Update
                x = x + v * dt
                
            generated_images.append(x.cpu())
            titles.append(f"Steps: {steps}")

    # 2. Visualization
    # Concatenate all lists: [16 imgs, 16 imgs, ...]
    
    # Create a grid: Rows = Step counts, Cols = Samples
    # But make_grid takes a flat list.
    
    # Let's just save separate grids or one big one.
    # Layout:
    # Row 1: 1-step
    # Row 2: 2-step
    # ...
    
    final_grid_parts = []
    for imgs in generated_images:
        # imgs is [16, 3, H, W]
        # Normalize to [0, 1] for saving
        # Assuming Data was normalized ? If VAE, need decode.
        # Assuming pixel space [-1, 1] for now or N(0,1)
        # Simple Min-Max norm per image for vis
        imgs = torch.clamp(imgs, -3, 3) 
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        
        grid = make_grid(imgs, nrow=8, padding=2)
        final_grid_parts.append(grid)
        
    # Save Images
    if args.save_dir:
        save_path = os.path.join(args.save_dir, 'samples')
        os.makedirs(save_path, exist_ok=True)
        
        for idx, (grid, title) in enumerate(zip(final_grid_parts, titles)):
            grid_np = grid.permute(1, 2, 0).numpy()
            plt.figure(figsize=(10, 5))
            plt.imshow(grid_np)
            plt.axis('off')
            plt.title(f"Step {step} | {title}")
            plt.savefig(os.path.join(save_path, f"step_{step}_{title.replace(' ', '')}.png"))
            plt.close()
            
    print("Evaluation Complete.")
    model.train()
