from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Callable, List

import numpy as np

from experiment_utils.utils import ensure_dir


def _standardize_frame(
    frame: Any,
    *,
    target_size: Optional[Tuple[int, int]] = None,  # (H, W)
    bgr_to_rgb: bool = False,
) -> Optional[np.ndarray]:
    """Convert a render() result to uint8 np.ndarray [H,W,C] with optional resize."""
    if frame is None:
        return None

    # to numpy
    if not isinstance(frame, np.ndarray):
        try:
            frame = np.asarray(frame)
        except Exception:
            return None

    # float -> uint8
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(
            np.uint8
        )
    elif frame.dtype != np.uint8:
        try:
            frame = frame.astype(np.uint8)
        except Exception:
            pass

    # grayscale -> 3-ch
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)

    # BGRA/BGR -> RGB (optional)
    if bgr_to_rgb and frame.ndim == 3 and frame.shape[-1] >= 3:
        frame = frame[..., :3][:, :, ::-1]

    # resize (if requested)
    if target_size is not None:
        try:
            from PIL import Image  # lazy import, optional

            img = Image.fromarray(frame)
            frame = np.asarray(
                img.resize((int(target_size[1]), int(target_size[0])), Image.BILINEAR)
            )
        except Exception:
            # fallback: keep original if PIL missing
            pass

    return frame


def save_policy_media(
    *,
    model: Any,  # SB3-like model with .predict(obs, deterministic=...)
    env: Any,  # Gymnasium single-env (not VecEnv)
    out_path: str,  # where to save, suffix decides format if fmt=None
    episodes: int = 3,
    start_seed: int = 12345,
    max_steps: int = 200,
    deterministic: bool = True,
    fps: int = 8,
    fmt: Optional[str] = None,  # "gif" | "mp4" | None -> infer from out_path
    fix_render_mode: bool = True,  # set env.render_mode="rgb_array" if available
    target_size: Optional[Tuple[int, int]] = None,  # (H,W) resize each frame
    bgr_to_rgb: bool = False,  # set True if your renderer returns BGR
    render_each_step: bool = True,  # if False, only record frames at reset/terminal (usually keep True)
    pre_step_hook: Optional[Callable[[Any, Any], None]] = None,  # (env, model) -> None
    post_step_hook: Optional[
        Callable[[Any, Any, dict], None]
    ] = None,  # (env, model, info) -> None
    close_env: bool = False,  # whether to env.close() at the end
) -> Optional[str]:
    """
    Roll out a policy and save frames as GIF/MP4. Returns saved path or None.
    Assumptions:
      - `env` is a single Gymnasium env (not VecEnv); observation is compatible with model.predict.
      - `env.render()` returns an RGB(A) array; use `fix_render_mode=True` to try enabling rgb_array.

    Notes:
      - If you want smaller files, lower `fps`, reduce `episodes`/`max_steps`, or set `target_size`.
      - If your renderer returns BGR (e.g., via cv2), set `bgr_to_rgb=True`.
    """
    try:
        import imageio.v2 as imageio  # writer
    except Exception as e:
        print(f"[media] imageio not available: {e}")
        return None

    # Try to force rgb_array
    if fix_render_mode and hasattr(env, "render_mode"):
        try:
            env.render_mode = "rgb_array"
        except Exception:
            pass

    ensure_dir(os.path.dirname(out_path))
    frames: List[np.ndarray] = []
    first_frame_shape: Optional[Tuple[int, int]] = None

    try:
        for ep in range(int(episodes)):
            obs, _info = env.reset(seed=int(start_seed) + ep)

            # initial frame
            try:
                fr = env.render() if render_each_step else None
            except Exception:
                fr = None
            fr = _standardize_frame(fr, target_size=target_size, bgr_to_rgb=bgr_to_rgb)
            if fr is not None:
                frames.append(fr)
                first_frame_shape = first_frame_shape or fr.shape[:2]

            # run loop
            for t in range(int(max_steps)):
                if callable(pre_step_hook):
                    try:
                        pre_step_hook(env, model)
                    except Exception:
                        pass

                action, _state = model.predict(obs, deterministic=bool(deterministic))
                obs, reward, terminated, truncated, info = env.step(action)

                if render_each_step:
                    try:
                        fr = env.render()
                    except Exception:
                        fr = None
                    fr = _standardize_frame(
                        fr, target_size=target_size, bgr_to_rgb=bgr_to_rgb
                    )
                    if fr is not None:
                        # normalize size to first frame to avoid writer issues
                        if (first_frame_shape is not None) and fr.shape[
                            :2
                        ] != first_frame_shape:
                            fr = _standardize_frame(
                                fr, target_size=first_frame_shape, bgr_to_rgb=False
                            )
                        frames.append(fr)

                if callable(post_step_hook):
                    try:
                        post_step_hook(
                            env, model, info if isinstance(info, dict) else {}
                        )
                    except Exception:
                        pass

                if terminated or truncated:
                    # final frame at episode end
                    try:
                        fr = env.render()
                    except Exception:
                        fr = None
                    fr = _standardize_frame(
                        fr, target_size=target_size, bgr_to_rgb=bgr_to_rgb
                    )
                    if fr is not None:
                        if (first_frame_shape is not None) and fr.shape[
                            :2
                        ] != first_frame_shape:
                            fr = _standardize_frame(
                                fr, target_size=first_frame_shape, bgr_to_rgb=False
                            )
                        frames.append(fr)
                    break

        if not frames:
            print("[media] no frames captured; skip.")
            return None

        # Decide format
        fmt = (fmt or os.path.splitext(out_path)[1].lstrip(".") or "gif").lower()
        if fmt == "gif":
            # duration: seconds per frame
            imageio.mimsave(out_path, frames, duration=max(1e-6, 1.0 / float(fps)))
        else:
            writer = imageio.get_writer(out_path, fps=int(fps))
            for fr in frames:
                writer.append_data(fr)
            writer.close()

        print(f"[media] saved to {out_path} ({len(frames)} frames)")
        return out_path
    except Exception as e:
        print(f"[media] failed to save media: {e}")
        return None
    finally:
        if close_env:
            try:
                env.close()
            except Exception:
                pass
