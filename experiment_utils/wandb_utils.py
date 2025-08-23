from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import ray
import wandb

from experiment_utils.utils import now_tag


def wandb_init(args) -> "wandb.sdk.wandb_run.Run":
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        print("[W&B] Offline mode.")
    elif args.wandb_mode != "online":
        raise ValueError("--wandb-mode must be 'online' or 'offline'")

    name = args.run_name or f"fullexp_{args.map}_{'slip' if args.slippery else 'noslip'}_" \
           f"ph{'-'.join(str(x) for x in args.phase_steps)}_seeds{len(args.train_seeds)}_{now_tag()}"

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


def wandb_log_image(run, key: str, path: Path):
    run.log({key: wandb.Image(str(path))})


@ray.remote
class WandbWriter:
    """
    Thin wrapper over wandb. Create this actor OUTSIDE and pass its handle into
    Collector/Driver. If W&B unavailable or init_kwargs is None, it becomes a no-op.
    """
    def __init__(self, init_kwargs: Optional[Dict[str, Any]] = None):
        self._on = False
        self._run = None
        if init_kwargs:
            try:
                self._run = wandb.init(**init_kwargs)
                self._on = True
            except Exception as e:
                print(f"[W&B] init failed: {e}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self._on: return
        try:
            self._run.log(data, step=step)
        except Exception as e:
            print(f"[W&B] log failed: {e}")

    def log_video(self, key: str, path: str, fps: int = 8, fmt: Optional[str] = None):
        if not self._on: return
        try:
            if (fmt or "").lower() == "gif" or str(path).lower().endswith(".gif"):
                self._run.log({key: wandb.Video(path, fps=int(fps), format="gif")})
            else:
                self._run.log({key: wandb.Video(path, fps=int(fps))})
        except Exception as e:
            print(f"[W&B] video log failed ({key}): {e}")

    def log_image(self, key: str, path: str):
        if not self._on: return
        try:
            self._run.log({key: wandb.Image(path)})
        except Exception as e:
            print(f"[W&B] image log failed ({key}): {e}")

    def finish(self):
        if not self._on: return
        try:
            self._run.finish()
        except Exception:
            pass
