import torch
import numpy as np
import os
import tqdm

def do_inference(args, model, step, device='cuda'):
    """
    Stand-alone inference script to generate a batch of images and save them.
    Adapted from JAX implementation to PyTorch.
    """
    print(f"--- Inference at Step {step} ---")
    
    # 1. Config
    # Check if we have specific inference args, otherwise fallback to defaults
    inference_timesteps = getattr(args, 'inference_timesteps', 128)
    inference_generations = getattr(args, 'inference_generations', 16) # Reduced default from 4096 for quick check
    cfg_scale = getattr(args, 'inference_cfg_scale', 4.0)
    batch_size = getattr(args, 'batch_size', 16)
    
    num_classes = args.model_config.num_classes
    img_size = args.model_config.image_size
    channels = args.model_config.image_channels
    
    save_dir = args.save_dir if args.save_dir else "results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Sampling {inference_generations} images with {inference_timesteps} steps, CFG={cfg_scale}")
    
    model.eval()
    
    all_images = []
    
    num_batches = int(np.ceil(inference_generations / batch_size))
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches)):
            current_batch_size = min(batch_size, inference_generations - i * batch_size)
            
            # 2. Setup Noise and Labels
            x = torch.randn(current_batch_size, channels, img_size, img_size, device=device)
            y = torch.randint(0, num_classes, (current_batch_size,), device=device)
            
            # 3. Sampling Loop (Euler)
            delta_t = 1.0 / inference_timesteps
            
            for ti in range(inference_timesteps):
                t_val = ti / inference_timesteps
                t = torch.ones(current_batch_size, device=device) * t_val
                
                # Determine dt_base index
                # If naive flow matching, we might use log2(T)
                # If shortcut, we use the specific index for this speed
                dt_k = int(np.log2(inference_timesteps))
                dt_base = torch.ones(current_batch_size, device=device, dtype=torch.long) * dt_k
                
                if cfg_scale == 0:
                    # Unconditional
                    # Need to pass null token. Assuming LabelEmbedder handles dropping if we pass a flag,
                    # or we pass the null class ID manually.
                    # PyTorch implementation of LabelEmbedder expected 'force_drop_ids'.
                    # Let's pass force_drop_ids=1 to drop.
                   
                    # Calling model: forward(self, x, t, dt, y, train=False, return_activations=False)
                    # We need to manually handle the dropout inside or pass correct args.
                    # Our LabelEmbedder `forward` takes `force_drop_ids`. 
                    # But `DiT.forward` usually calls `y_embedder` with `y`.
                    
                    # Hack: The DiT.forward we modified expects y. 
                    # The LabelEmbedder logic we pasted: forward(labels, train, force_drop_ids)
                    # But DiT.forward calls: y_emb = self.y_embedder(y, self.training)
                    # It DOES NOT pass force_drop_ids down.
                    
                    # Fix: Pass y = num_classes (which is the null token index)
                    y_null = torch.ones_like(y) * num_classes
                    v = model(x, t, dt_base, y_null, train=False)
                    
                elif cfg_scale == 1:
                    # Conditional only
                    v = model(x, t, dt_base, y, train=False)
                    
                else:
                    # CFG
                    # Cond
                    v_cond = model(x, t, dt_base, y, train=False)
                    
                    # Uncond (Null Token)
                    y_null = torch.ones_like(y) * num_classes
                    v_uncond = model(x, t, dt_base, y_null, train=False)
                    
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                
                # Euler Update
                x = x + v * delta_t
                
            # 4. Collect
            all_images.append(x.cpu().numpy())
            
    # 5. Save
    all_images = np.concatenate(all_images, axis=0) # [N, C, H, W]
    
    # Save as numpy array
    np.save(os.path.join(save_dir, f"generated_samples_step_{step}.npy"), all_images)
    print(f"Saved to {os.path.join(save_dir, f'generated_samples_step_{step}.npy')}")
    
    model.train()
