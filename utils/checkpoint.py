import torch
import os
import shutil

class Checkpoint:
    """
    Simulated Checkpoint class matching the API expected by Train.py
    but using PyTorch primitives.
    """
    def __init__(self, filename=None, parallel=False):
        self._filename = filename
        
    def save(self, save_dir, step, model, optimizer=None, args=None, extra_state=None):
        """
        Saves a checkpoint to save_dir/checkpoint_{step}.pt
        """
        if not save_dir:
            print("No save_dir specified, skipping checkpoint.")
            return

        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"checkpoint_{step}.pt")
        
        # Construct the state dict
        state = {
            'step': step,
            'model': model.state_dict(),
        }
        
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
            
        if args is not None:
            # Save config if available
            # Convert namespace to dict if needed
            if hasattr(args, '__dict__'):
                state['args'] = args.__dict__
            else:
                state['args'] = args

        if extra_state is not None:
            state.update(extra_state)

        # Atomic save
        tmp_filename = filename + ".tmp"
        torch.save(state, tmp_filename)
        shutil.move(tmp_filename, filename)
        
        print(f"Saved checkpoint to {filename}")
        
    def load(self, load_dir, model, optimizer=None, device='cpu'):
        """
        Loads the latest checkpoint from load_dir.
        """
        if not load_dir or not os.path.exists(load_dir):
            print(f"Checkpoint directory {load_dir} does not exist.")
            return 0 # Start step

        # Find latest checkpoint
        # Assuming format checkpoint_STEP.pt
        files = [f for f in os.listdir(load_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not files:
            print(f"No checkpoints found in {load_dir}")
            return 0
            
        # Parse steps
        steps = []
        for f in files:
            try:
                s = int(f.split('_')[1].split('.')[0])
                steps.append(s)
            except:
                pass
        
        if not steps:
            return 0
            
        latest_step = max(steps)
        filename = os.path.join(load_dir, f"checkpoint_{latest_step}.pt")
        
        print(f"Loading checkpoint {filename}...")
        
        checkpoint = torch.load(filename, map_location=device)
        
        # Load Model
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict']) # Compat
        else:
            print("Warning: No model state found in checkpoint.")

        # Load Optimizer
        if optimizer is not None:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif 'optimizer_state_dict' in checkpoint: # Compat
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
                
        print(f"Loaded successfully from step {latest_step}")
        return latest_step
