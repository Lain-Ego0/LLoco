"""Go2 action terms."""

from dataclasses import dataclass

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.buffers import DelayBuffer


class EpisodeDelayedJointPositionAction(JointPositionAction):
  """One shared 1--3 physics-substep command lag, sampled per environment."""

  def __init__(self, cfg: "EpisodeDelayedJointPositionActionCfg", env) -> None:
    super().__init__(cfg, env)
    self._source_delay = DelayBuffer(
      min_lag=cfg.delay_min_lag,
      max_lag=cfg.delay_max_lag,
      batch_size=env.num_envs,
      device=env.device,
      per_env=True,
      update_period=cfg.delay_update_period,
      per_env_phase=False,
    )

  def apply_actions(self) -> None:
    self._source_delay.append(self._processed_actions)
    target = self._source_delay.compute()
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    self._source_delay.reset(batch_ids=env_ids)


@dataclass(kw_only=True)
class EpisodeDelayedJointPositionActionCfg(JointPositionActionCfg):
  delay_min_lag: int = 1
  delay_max_lag: int = 3
  delay_update_period: int = 2**30

  def build(self, env) -> EpisodeDelayedJointPositionAction:
    return EpisodeDelayedJointPositionAction(self, env)
