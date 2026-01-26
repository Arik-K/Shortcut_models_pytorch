import ml_collections

def default_wandb_config():
    """
    Returns a dummy config object to satisfy imports.
    """
    config = ml_collections.ConfigDict()
    config.project = "shortcut"
    config.name = "default"
    return config

def setup_wandb(hyperparam_dict, **kwargs):
    """
    Dummy setup_wandb that does nothing.
    """
    print("WandB is disabled. Logging will be skipped.")
    
    # Return a dummy object that has a .log method so calls don't crash
    class DummyRun:
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass
            
    return DummyRun()

# Monkey-patch wandb module so direct wandb.log() calls also don't crash if used
class DummyWandbModule:
    def log(self, *args, **kwargs):
        pass
    def init(self, *args, **kwargs):
        return DummyWandbModule()
    def finish(self):
        pass
    
    # Add other common wandb methods if needed
    Image = lambda x: x # Mock Image wrapper
    
import sys
# If 'wandb' is not installed, we can mock it entirely in sys.modules
# But Train.py imports "import wandb" on line 7.
# If the user has wandb installed, we just don't init it.
# If they don't have it installed, we might need to mock it.
