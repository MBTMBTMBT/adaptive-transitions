from __future__ import annotations

import os
import sys
import threading
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

    name = (
        args.run_name
        or f"fullexp_{args.map}_{'slip' if args.slippery else 'noslip'}_"
        f"ph{'-'.join(str(x) for x in args.phase_steps)}_seeds{len(args.train_seeds)}_{now_tag()}"
    )

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


@ray.remote  # single-threaded actor by default -> serialized W&B calls
class WandbActor:
    def __init__(
        self, init_kwargs: Dict[str, Any], env: Dict[str, str] | None = None
    ) -> None:
        # Safer start method in multi-proc envs; optional overrides via env.
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        if env:
            for k, v in env.items():
                os.environ[k] = v

        self.wandb = wandb
        self.run = wandb.init(**init_kwargs)

    # -------- Common helpers --------
    def log(self, metrics: Dict[str, Any], **kwargs) -> None:
        self.run.log(dict(metrics), **kwargs)

    def summary_update(self, d: Dict[str, Any], **kwargs) -> None:
        self.run.summary.update(dict(d), **kwargs)

    def config_update(self, d: Dict[str, Any], **kwargs) -> None:
        # allow_val_change etc. can be passed in kwargs
        self.run.config.update(dict(d), **kwargs)

    def define_metric(self, *args, **kwargs) -> None:
        self.wandb.define_metric(*args, **kwargs)

    def log_artifact_dir(
        self,
        name: str,
        a_type: str,
        dir_path: str,
        metadata: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        art = self.wandb.Artifact(name=name, type=a_type, metadata=(metadata or {}))
        art.add_dir(dir_path)
        self.run.log_artifact(art, **kwargs)

    def log_image(
        self, key: str, path: str | os.PathLike, caption: str | None = None, **kwargs
    ) -> None:
        img = self.wandb.Image(str(path), caption=caption)
        self.run.log({key: img}, **kwargs)

    def log_video(
        self,
        key: str,
        path: str | os.PathLike,
        fps: int | None = None,
        fmt: str | None = None,
        **kwargs,
    ) -> None:
        vkw = {}
        if fps is not None:
            vkw["fps"] = int(fps)
        if fmt is not None:
            vkw["format"] = str(fmt)  # W&B uses 'format' param
        vid = self.wandb.Video(str(path), **vkw)
        self.run.log({key: vid}, **kwargs)

    # -------- Generic passthroughs to cover "latest W&B features" --------
    def call_run(self, method: str, *args, **kwargs) -> Any:
        # Example: call_run("watch", model, log="all", log_freq=100)
        return getattr(self.run, method)(*args, **kwargs)

    def call_wandb(self, method: str, *args, **kwargs) -> Any:
        # Example: call_wandb("alert", title="x", text="y")
        return getattr(self.wandb, method)(*args, **kwargs)

    def write_console(self, text: str, stream: str = "stdout") -> None:
        """Forward a line into this actor's console, which W&B Logs captures."""
        import sys

        s = str(text)
        # Ensure line break to avoid sticking lines together
        if not (s.endswith("\n") or s.endswith("\r")):
            s += "\n"
        if stream == "stderr":
            sys.stderr.write(s)
            sys.stderr.flush()
        else:
            sys.stdout.write(s)
            sys.stdout.flush()

    def finish(self) -> None:
        self.run.finish()


def capture_prints_to_wandb(wandb_actor, capture_stderr: bool = True) -> None:
    """
    Tee sys.stdout/sys.stderr to the WandbActor so all prints appear in W&B Logs.
    Non-blocking per line: uses actor.write_console.remote().
    """

    class _Tee:
        def __init__(self, original, stream_name: str):
            self._orig = original
            self._stream_name = stream_name
            self._buf = ""
            self._lock = threading.Lock()

        def write(self, s: str) -> int:
            with self._lock:
                written = self._orig.write(s)  # keep local console behavior
                self._orig.flush()
                self._buf += s
                # Split by newline or carriage return to forward complete lines
                while ("\n" in self._buf) or ("\r" in self._buf):
                    sep = "\n" if "\n" in self._buf else "\r"
                    line, self._buf = self._buf.split(sep, 1)
                    if line:
                        # Async, one RPC per completed line
                        wandb_actor.write_console.remote(line, stream=self._stream_name)
                return written

        def flush(self) -> None:
            self._orig.flush()

    sys.stdout = _Tee(sys.stdout, "stdout")
    if capture_stderr:
        sys.stderr = _Tee(sys.stderr, "stderr")
