def below_reset_height(env, reset_height=0.1):
  return env.scene["robot"].data.root_link_pos_w[:, 2] <= reset_height
