from customised_minigrid_env import CustomMiniGridEnv
from customised_minigrid_env.simple_manual_control import SimpleManualControl

if __name__ == "__main__":
    env = CustomMiniGridEnv(
        map_name="door-key-fixed",
        config=None,
        display_size=None,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        render_carried_objs=True,
        render_mode="human",
    )
    manual_control = SimpleManualControl(env)
    manual_control.start()
