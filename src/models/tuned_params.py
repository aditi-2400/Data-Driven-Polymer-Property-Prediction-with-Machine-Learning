import json, os
from typing import Dict

def load_tuned_params(path: str) -> Dict:
    if not os.path.exists(path): return {}
    with open(path, "r") as f:
        return json.load(f)
