from __future__ import annotations
from importlib.resources import files, as_file
from contextlib import contextmanager
from pathlib import Path
import json
from typing import Dict, List

# Anchor to the package root; we treat "maps" as a data directory, not a subpackage.
_ROOT_PKG = __name__
_MAPS_DIR = files(_ROOT_PKG) / "maps"

# Friendly names -> resource filenames (inside customised_minigrid_env/maps)
MAPS: Dict[str, str] = {
    "door-key": "door-key.json",
    "door-key-fixed": "door-key-fixed.json",
    "3-rooms-2-doors-2-keys": "three-rooms-two-doors-two-keys.json",
    "2-doors-2-keys": "two-doors-two-keys.json",
}

def list_builtin_maps() -> List[str]:
    """Return the list of available built-in map names (keys of MAPS)."""
    return sorted(MAPS.keys())

def _resource_for(name: str):
    """Return a Traversable resource for the given map name."""
    try:
        return _MAPS_DIR / MAPS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown map '{name}'. Available: {list_builtin_maps()}"
        ) from e

def load_map_config_by_name(name: str) -> dict:
    """Load the JSON config from packaged resources by 'name'."""
    res = _resource_for(name)
    return json.loads(res.read_text(encoding="utf-8"))

@contextmanager
def open_map_path(name: str):
    """
    Yield a real filesystem Path for tools that require a file path.
    Works whether the package is installed as files or zipped.
    """
    res = _resource_for(name)
    with as_file(res) as real_path:
        yield Path(real_path)

# Re-export your env class for convenient imports like:
#   from customised_minigrid_env import CustomMiniGridEnv
from .customised_minigrid_env import CustomMiniGridEnv
