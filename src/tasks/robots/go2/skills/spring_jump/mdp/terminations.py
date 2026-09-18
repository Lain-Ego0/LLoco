"""Additional source termination for the spring-jump task."""


def below_reset_height(env, reset_height: float = 0.15):
  return env.scene["robot"].data.root_link_pos_w[:, 2] <= reset_height
