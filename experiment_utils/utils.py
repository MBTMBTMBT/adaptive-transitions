from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import List, Tuple, Any

import wandb


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y", "on")


def _parse_csv_numbers(s: str, typ=float) -> List:
    if s is None or s == "":
        return []
    return [typ(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_tuple3(s: str) -> Tuple[float, float, float]:
    parts = _parse_csv_numbers(s, float)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated floats, e.g. '1.0,0.0,0.0'")
    return tuple(parts)  # type: ignore


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, obj: Any):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _wandb_init(args) -> "wandb.sdk.wandb_run.Run":
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        print("[W&B] Offline mode.")
    elif args.wandb_mode != "online":
        raise ValueError("--wandb-mode must be 'online' or 'offline'")

    name = args.run_name or f"fullexp_{args.map}_{'slip' if args.slippery else 'noslip'}_" \
           f"ph{'-'.join(str(x) for x in args.phase_steps)}_seeds{len(args.train_seeds)}_{_now_tag()}"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=name,
        job_type="full-exp",
        config={},  # updated below
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    cfg = {k: getattr(args, k) for k in vars(args)}
    run.config.update(cfg, allow_val_change=True)
    return run


def _wandb_log_image(run, key: str, path: Path):
    run.log({key: wandb.Image(str(path))})


def _resolve_args(p: argparse.ArgumentParser) -> argparse.Namespace:
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
        args.train_seeds = [int(x) for x in _parse_csv_numbers(ts, int)]

    # phase steps: CSV -> list[int], need >= 2
    args.phase_steps = [int(x) for x in _parse_csv_numbers(args.phase_steps, int)]
    if len(args.phase_steps) < 2:
        raise SystemExit("--phase-steps requires at least 2 phases (Source then Target).")

    _ensure_dir(Path(args.outdir) / "meta")
    return args
