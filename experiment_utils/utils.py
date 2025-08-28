from __future__ import annotations

import argparse
import datetime
import importlib
import json
from pathlib import Path
from typing import List, Tuple, Any, Union


def ensure_dir(p: Union[str, Path]) -> Path:
    """
    Accept str or pathlib.Path; always return a pathlib.Path.
    """
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y", "on")


def parse_csv_numbers(s: str, typ=float) -> List:
    if s is None or s == "":
        return []
    return [typ(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_tuple3(s: str) -> Tuple[float, float, float]:
    parts = parse_csv_numbers(s, float)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected 3 comma-separated floats, e.g. '1.0,0.0,0.0'"
        )
    return tuple(parts)  # type: ignore


def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, obj: Any):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def resolve_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()

    # agent kwargs: assembled from flattened CLI flags
    args.agent_kwargs = dict(
        learning_rate=args.agent_learning_rate,
        gamma=args.agent_gamma,
        policy_mix=tuple(args.agent_policy_mix),
        temperature=args.agent_temperature,
        tie_tol=args.agent_tie_tol,
        verbose=args.agent_verbose,
    )

    # train-seeds: COUNT N -> [0..N-1]
    if isinstance(args.train_seeds, int):
        args.train_seeds = list(range(args.train_seeds))
    else:
        # fallback if someone passes CSV by mistake
        ts = str(args.train_seeds).strip()
        args.train_seeds = [int(x) for x in parse_csv_numbers(ts, int)]

    # phase steps: CSV -> list[int], need >= 2
    args.phase_steps = [int(x) for x in parse_csv_numbers(args.phase_steps, int)]
    if len(args.phase_steps) < 2:
        raise SystemExit(
            "--phase-steps requires at least 2 phases (Source then Target)."
        )

    ensure_dir(Path(args.outdir) / "meta")
    return args


def _import(path: str):
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)

def _timestamped_outdir(base_outdir: str, leaf: str) -> Path:
    """
    Wrap the user-provided outdir as:
        <base_outdir>/<leaf>/<YYYYmmdd-HHMMSS>
    """
    base = Path(base_outdir).expanduser().resolve()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / leaf / ts
