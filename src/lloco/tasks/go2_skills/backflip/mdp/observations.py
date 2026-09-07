"""Exact source 47 actor / 50 privileged observation frames."""
import torch
from mjlab.entity import Entity
from ...jump.mdp.observations import _SourceHistory, joint_ids

def _gravity(robot): return robot.data.projected_gravity_b
def _actor(env, command_name):
  r: Entity = env.scene["robot"]; ids = joint_ids(r); c = env.command_manager.get_command(command_name); assert c is not None
  return torch.cat((torch.zeros((env.num_envs,2), device=env.device),c,r.data.root_link_ang_vel_b*.25,_gravity(r),r.data.joint_pos[:,ids]-r.data.default_joint_pos[:,ids],r.data.joint_vel[:,ids]*.05,env.action_manager.action),1)
def _critic(env, command_name):
  r: Entity = env.scene["robot"]; ids = joint_ids(r); c = env.command_manager.get_command(command_name); assert c is not None
  return torch.cat((torch.zeros((env.num_envs,2),device=env.device),c,r.data.joint_pos[:,ids]-r.data.default_joint_pos[:,ids],r.data.joint_vel[:,ids]*.05,env.action_manager.action,r.data.root_link_lin_vel_b*2.,r.data.root_link_ang_vel_b*.25,_gravity(r)),1)
class BackflipActorHistory(_SourceHistory):
  frame_dim=47; history_length=10
  def __call__(self,env,command_name,add_noise):
    x=_actor(env,command_name)
    if add_noise:
      a=torch.tensor((0.,)*5+(.05,)*3+(.1,)*3+(.01,)*12+(.075,)*12+(0.,)*12,device=env.device); x=x+(2*torch.rand_like(x)-1)*a
    return self._append(x)
class BackflipCriticHistory(_SourceHistory):
  frame_dim=50; history_length=3
  def __call__(self,env,command_name): return self._append(_critic(env,command_name))
