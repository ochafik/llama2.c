from pathlib import Path
from safetensors.torch import save_file
from jsonargparse import CLI
import torch
import re
import json

def main(folder_in: Path, folder_out: Path) -> None:
    """Converts a pickle-serialized model to safetensors"""

    mapping = {}

    for pickle_in in folder_in.glob('pytorch_model-*.bin'):
        safe_out = (folder_out / re.sub(r'pytorch_model-(\d+)-of-(\d+)\.bin$', r'model-\1-of-\2.safetensors', pickle_in.name))
        mapping[pickle_in.name] = safe_out.name
        
        if safe_out.exists():
            continue
        print(f"Converting {pickle_in} to {safe_out}")
        save_file(torch.load(pickle_in, map_location='cpu'), safe_out)
    
    for json_in in folder_in.glob('*.json'):
        if json_in.name == 'pytorch_model.bin.index.json':
            with json_in.open('r') as f:
                index = json.load(f)
            with (folder_out / 'model.safetensors.index.json').open('w') as f:
                json.dump({k: mapping[v] for k, v in index.items()}, f, indent=2)
        else:
            json_in.copy(folder_out / json_in.name)

if __name__ == "__main__":
    CLI(main)
