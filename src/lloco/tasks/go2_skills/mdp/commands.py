"""Go2 command generators."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import viser
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class TrotVelocityCommand(UniformVelocityCommand):
  """Source sampling: 5% all-zero, then an independent 5% XY-zero draw."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    all_zero = torch.rand(len(env_ids), device=self.device) < 0.05
    self.vel_command_b[env_ids[all_zero]] = 0.0
    xy_zero = torch.rand(len(env_ids), device=self.device) < 0.05
    self.vel_command_b[env_ids[xy_zero], :2] = 0.0
    moving_xy = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) > 0.1
    self.vel_command_b[env_ids, :2] *= moving_xy.unsqueeze(1)


@dataclass(kw_only=True)
class TrotVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> TrotVelocityCommand:
    return TrotVelocityCommand(self, env)


class RearStandVelocityCommand(UniformVelocityCommand):
  """Gym ``go2_handstand`` sampling and heading controller for Rear Stand."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    all_zero = torch.rand(len(env_ids), device=self.device) < 0.20
    zero_ids = env_ids[all_zero]
    self.vel_command_b[zero_ids] = 0.0
    self.heading_target[zero_ids] = 0.0
    xy_zero = torch.rand(len(env_ids), device=self.device) > 0.90
    self.vel_command_b[env_ids[xy_zero], :2] = 0.0
    moving_xy = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) > 0.1
    self.vel_command_b[env_ids, :2] *= moving_xy.unsqueeze(1)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # The Gym task applies heading control to every environment and clips to
    # [-1, 1], independently of its nominal sampled yaw range.
    del env_ids
    self.heading_error = torch.atan2(
      torch.sin(self.heading_target - self.robot.data.heading_w),
      torch.cos(self.heading_target - self.robot.data.heading_w),
    )
    self.vel_command_b[:, 2] = torch.clamp(0.5 * self.heading_error, -1.0, 1.0)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create a joystick that supports the source's fixed zero Y range."""
    del on_change, request_action
    from viser import Icon

    axes = (
      ("lin_vel_x", self.cfg.ranges.lin_vel_x[1]),
      ("lin_vel_y", self.cfg.ranges.lin_vel_y[1]),
      ("ang_vel_z", self.cfg.ranges.ang_vel_z[1]),
    )
    sliders = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        if max_val == 0.0:
          slider = server.gui.add_slider(
            label,
            min=0.0,
            max=0.0,
            step=0.05,
            initial_value=0.0,
            disabled=True,
          )
          sliders.append(slider)
          continue

        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx


@dataclass(kw_only=True)
class RearStandVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> RearStandVelocityCommand:
    return RearStandVelocityCommand(self, env)
