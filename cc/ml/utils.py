import torch

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, 
                    map_location: str | torch.device = "cpu", weights_only:bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=False)