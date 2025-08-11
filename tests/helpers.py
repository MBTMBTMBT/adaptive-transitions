from typing import Optional, List, Callable

from PIL import Image, ImageDraw


def _overlay_label(frame_ndarray, text: str) -> Optional[Image.Image]:
    """Overlay a small black bar with text on top of an RGB frame."""
    if frame_ndarray is None:
        return None
    img = Image.fromarray(frame_ndarray)
    draw = ImageDraw.Draw(img)
    bar_h = 22
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255))
    return img


def record_policy_gif(
    env,
    policy,
    action_names: List[str],
    seed: int,
    episodes: int,
    max_steps: int,
    *,
    use_encode: bool = False,
    state_fn: Optional[Callable] = None,
) -> List[Image.Image]:
    """
    Run episodes following a (probabilistic) policy and collect frames.

    Notes:
    - For Taxi/FrozenLake: obs is already an integer state id -> keep use_encode=False (default).
    - For MiniGrid (dict obs): pass use_encode=True or provide state_fn to return an int id
      (e.g., lambda env, obs: env.encode_state()).
    """
    frames: List[Image.Image] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        # Derive the state id for the policy
        if state_fn is not None:
            state = state_fn(env, obs)
        elif use_encode and hasattr(env, "encode_state"):
            state = env.encode_state()
        else:
            state = obs  # works for Taxi/FrozenLake

        # First frame
        frame = env.render()
        img = _overlay_label(frame, f"state={state}, action=-")
        if img:
            frames.append(img)

        ep_reward = 0.0
        for t in range(max_steps):
            # Greedy action from policy distribution
            action_probs = policy.get_action_probabilities(state)
            action = max(action_probs.items(), key=lambda x: x[1])[0] if action_probs else env.action_space.sample()

            obs_next, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            frame = env.render()
            an = action_names[action] if 0 <= action < len(action_names) else str(action)
            img = _overlay_label(frame, f"state={state}, action={an}")
            if img:
                frames.append(img)

            # Update state id for the policy
            if state_fn is not None:
                state = state_fn(env, obs_next)
            elif use_encode and hasattr(env, "encode_state"):
                state = env.encode_state()
            else:
                state = obs_next

            if terminated or truncated:
                # Pad a few frames for nicer loops
                for _ in range(5):
                    if frames:
                        frames.append(frames[-1])
                break
    return frames
