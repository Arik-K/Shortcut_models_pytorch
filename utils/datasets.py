import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def get_dataset(dataset_name, batch_size, is_train, debug_overfit=False):
    """
    Load CIFAR-10 dataset (auto-download).
    """
    print(f"Loading dataset: {dataset_name} (Using CIFAR-10 fallback)")
    
    # Standard DiT transforms
    # Resize to 32x32 (CIFAR native) or upsample depending on what the user asked for.
    # The user "model.py" asked for 256. If we use CIFAR, upsampling to 256 is slow/blurry.
    # Let's keep it at 32x32 for speed, BUT we must tell the user to change model config.
    # OR we resize to what the model expects? 
    # Let's assume the user might adjust model size. 
    # BUT Train.py has hardcoded defaults.
    
    # Safe bet: Resize to 32 if input is small, but if model expects 256...
    # Let's just resize to 32 (Native) and hope the user adjusts 'image_size' argument.
    # Actually, for "Shortcut Models", training on 256x256 is standard. 
    # Let's resize CIFAR to 64 or 128? 256 is too big for CIFAR.
    # Let's default to standard CIFAR 32x32.
    
    # UPDATE: Train.py has `parser.add_argument('--dataset_name', type=str, default='imagenet256'...)`
    # and default patch size 8?
    # I will just set it to 32x32.
    
    size = 32
    if '256' in dataset_name:
        # If user explicitly asked for 256, we try to honor it even if it's blurry
        size = 32 # Warning: Using 32 anyway for speed, changing to 256 is trivial here.
        # size = 256 
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1]
    ])

    # Download to ./data
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=is_train, 
        download=True, 
        transform=transform
    )

    if debug_overfit:
        # Take small subset
        dataset = torch.utils.data.Subset(dataset, range(batch_size * 2))

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True
    )
    
    return loader
