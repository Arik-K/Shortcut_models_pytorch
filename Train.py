from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import wandb
import argparse
import ml_collections


from ml_collections import config_flags 
from functools import partial


from utils.wandb import setup_wandb, default_wandb_config
from utils.checkpoint import Checkpoint
from utils.datasets import get_dataset
from model import DiT
from helper_eval import eval_model
from helper_inference import do_inference

# Argument Parsing
parser = argparse.ArgumentParser(description='PyTorch training script')
parser.add_argument('--dataset_name', type=str, default='imagenet256', help='Dataset name.')
parser.add_argument('--load_dir', type=str, default=None, help='Load directory for parameters.')
parser.add_argument('--save_dir', type=str, default=None, help='Save directory for parameters/models.')
parser.add_argument('--fid_stats', type=str, default=None, help='FID stats file.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--log_interval', type=int, default=1000, help='Logging interval.')
parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval.')
parser.add_argument('--save_interval', type=int, default=1000, help='Save interval.')
parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size.')
parser.add_argument('--max_steps', type=int, default=5000, help='Maximum training steps.')
parser.add_argument('--debug_overfit', type=int, default=0, help='Debug overfitting flag.')
parser.add_argument('--mode', type=str, default='train', help='train or inference.')
args = parser.parse_args()

# Model Config
model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 64,  # change this!
    'patch_size': 2,    # change this!
    'depth': 2,         # change this!
    'num_heads': 2,     # change this!
    'mlp_ratio': 1,     # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 10,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 0,
    'use_stable_vae': 0,  # Disabled: StableVAE not implemented yet
    'sharding': 'dp',   # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut' # or naive.
})

# WandB Config
wandb_config = ml_collections.ConfigDict({
    'project': 'shortcut',
    'name': f'shortcut_{args.dataset_name}',
})

def main():
    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_count = torch.cuda.device_count()
        global_device_count = device_count 
        print(f"Using {device_count} CUDA devices.")
    else:
        device = torch.device('cpu')
        device_count = 1
        global_device_count = 1
        print("Using CPU.")
    print("Global Device Count:", global_device_count)

    # Batch sizes
    # NOTE: In PyTorch DDP, valid_batch_size usually stays same, 
    # but train is split. Assuming input args.batch_size is GLOBAL.
    local_batch_size = args.batch_size // global_device_count
    print("Global Batch: ", args.batch_size)
    print("Local Batch (per device): ", local_batch_size)

    # Create wandb logger
    is_master = True # Placeholder for single script
    if is_master and args.mode == 'train':
        setup_wandb(model_config.to_dict(), **wandb_config)
        pass 
        
    # Dataset
    train_loader = get_dataset(args.dataset_name, local_batch_size, True, args.debug_overfit)
    valid_loader = get_dataset(args.dataset_name, local_batch_size, False, args.debug_overfit)
    
    # Get example batch to infer shapes
    example_obs, example_labels = next(iter(train_loader))
    example_obs = example_obs.to(device)
    example_obs_shape = example_obs.shape

    vae = None
    if model_config.use_stable_vae:
        vae = StableVAE.create().to(device)
        vae.eval() # Set to eval mode for feature extraction context
        
        if 'latent' in args.dataset_name:
             pass 
        else:
            with torch.no_grad():
                # example_obs = vae.encode(example_obs) 
                pass
                
        # example_obs_shape = example_obs.shape

    if args.fid_stats is not None:
        # from utils.fid import get_fid_network, fid_from_stats
        # get_fid_activations = get_fid_network().to(device)
        # truth_fid_stats = np.load(args.fid_stats)
        pass
    else:
        get_fid_activations = None
        truth_fid_stats = None

    ###################################
    # Creating Model and put on devices.
    ###################################
    # Update config with inferred shapes
    model_config.image_channels = example_obs_shape[1] # C is dim 1 in NCHW
    model_config.image_size = example_obs_shape[2]     # H is dim 2 in NCHW

    dit_args = {
        'patch_size': model_config.patch_size,
        'hidden_size': model_config.hidden_size,
        'depth': model_config.depth,
        'num_heads': model_config.num_heads,
        'mlp_ratio': model_config.mlp_ratio,
        'out_channels': example_obs_shape[1], # Input/Output channels match
        'class_dropout_prob': model_config.class_dropout_prob,
        'num_classes': model_config.num_classes,
        'dropout': model_config.dropout,
        'ignore_dt': False if (model_config.train_type in ('shortcut', 'livereflow')) else True,
    }
    
    # Instantiate Model
    # Assuming DiT is a torch.nn.Module
    model = DiT(**dit_args).to(device)
    
    # Print parameter count (equivalent to tabulate)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer & Scheduler
    if model_config.use_cosine:
        # Note: PyTorch schedulers work on steps or epochs. 
        # Typically define optimizer first, then scheduler.
        pass 
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=model_config.lr, 
        betas=(model_config.beta1, model_config.beta2), 
        weight_decay=model_config.weight_decay
    )
    
    # Scheduler
    scheduler = None
    if model_config.use_cosine:
        # Linear warmup then cosine decay
        # You might need a custom LambdaLR or SequentialLR for strictly matching Optax warmup+cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    elif model_config.warmup > 0:
        # Placeholder for linear warmup
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=model_config.warmup)

    # EMA Setup (Placeholder)
    # from torch_ema import ExponentialMovingAverage
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema = None

    # Resume Training / Checkpoint
    start_step = 0
    if args.load_dir is not None:
        # if os.path.exists(os.path.join(args.load_dir, "checkpoint.pt")):
        #     checkpoint = torch.load(os.path.join(args.load_dir, "checkpoint.pt"), map_location=device)
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     start_step = checkpoint['step']
        #     print(f"Loaded model from step {start_step}")
        pass

    # Teacher Model (for distillation)
    model_teacher = None
    if model_config.train_type in ['progressive', 'consistency-distillation']:
        # Copy model to create teacher
        import copy
        model_teacher = copy.deepcopy(model)
        for param in model_teacher.parameters():
            param.requires_grad = False
        model_teacher.eval()

    # Labels for visualization
    imagenet_labels = None
    try:
        with open('data/imagenet_labels.txt', 'r') as f:
            imagenet_labels = f.read().splitlines()
    except FileNotFoundError:
        print("Warning: imagenet_labels.txt not found.")
    
    ###################################
    # Training Loop
    ###################################
    
    model.train()
    
    # Create iterators for step-based training
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    pbar = tqdm.tqdm(total=args.max_steps, initial=start_step)

    # --- SWITCHOVER: Using UNet + ShortcutDiffusionWrapper ---
    from denoising_diffusion_pytorch import Unet, ShortcutDiffusion
    
    # 1. Create Model (UNet)
    # Using small config for CIFAR/Debug as requested
    unet_model = Unet(
        dim=model_config.hidden_size, # 64
        channels=model_config.image_channels, # 3
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        # self_condition=True # Optional, often helps
    ).to(device)

    # 2. Create Wrapper (Handles Loss & Targets internally)
    diffusion_model = ShortcutDiffusion(
        model=unet_model,
        image_size=model_config.image_size, # 32
        max_discretization_steps=model_config.denoise_timesteps, # 128
        auto_normalize=False # We handle normalization in dataset
    ).to(device)
    
    model = diffusion_model # Make 'model' points to the wrapper for saving/loading
    
    # Update optimizer to target the new model parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=model_config.lr, 
        weight_decay=model_config.wd, 
        betas=(model_config.beta1, model_config.beta2)
    )

    for step in range(start_step + 1, args.max_steps + 1):
        # 1. Get Train Batch
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
            
        images = images.to(device)
        # UNet handles labels internally or via codebook? 
        # The provided Unet snippet takes (x, time, codebook=None, d=None).
        # It DOES NOT seem to have a class label embedding layer built-in like the DiT.
        # It relies on 'codebook' which seems to be dense conditioning (like weather/seg).
        # For CIFAR class conditioning, you usually add a label embedding.
        # FIX: For now, we run UNCONDITIONAL (ignore labels) or pass labels as codebook if logic allows.
        # The snippet suggests: x_ref_s = codebook[mask_shortcut]
        # We will pass NONE for now (Unconditional).
        labels = None 
        
        # 2. Train Step (Forward + Loss)
        optimizer.zero_grad()
        
        # The Wrapper handles splitting, target gen, and loss calc
        # input: img_clean, img_foggy (noise). 
        # But wait, ShortcutDiffusion.forward(img_clean, img_foggy)
        # We need to generate the "foggy" (noise) or does it do it?
        # Re-reading ShortcutDiffusion.forward:
        # 1. Sample d, t. 
        # 2. x_t = (1-t) * foggy + t * clean.
        # It interpolates between 'clean' and 'foggy'.
        # Usually 'foggy' = Random Noise.
        
        noise = torch.randn_like(images)
        loss_dict = model(images, noise, codebook=labels) # Returns dict
        
        loss = loss_dict['loss_total']
        
        loss.backward()
        
        # Gradient Clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        if ema:
            ema.update()

        # 4. Logging & Validation
        if step % args.log_interval == 0 or step == 1:
            pbar.set_description(f"Step {step} | Loss: {loss.item():.4f}")
            if is_master:
                # print(f"Step {step}: {loss_dict}")
                pass

        # 5. Evaluation
        if step % args.eval_interval == 0:
            # We need to adapt eval_model because 'model' is now the Wrapper
            # Wrapper has 'model.sample(x_foggy, num_steps)'
            # Let's do a quick quick sample inline
            model.eval()
            with torch.no_grad():
                sample_noise = torch.randn(16, 3, 32, 32, device=device)
                # Sample 1 step
                one_step = model.sample(sample_noise, num_steps=1)
                # Sample 8 steps
                eight_step = model.sample(sample_noise, num_steps=8)
                
                # Save just one step for speed
                # utils.save_image(one_step, f"results/sample_{step}.png")
            model.train()

        # 7. Checkpointing
        if step % args.save_interval == 0 and args.save_dir is not None:
             if is_master:
                save_path = f"{args.save_dir}/checkpoint_{step}.pt"
                # Save inner model state to be safe
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

        pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    main()