"""State machine and source reward equations for Gym Backflip."""
import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from ...shared.contacts import joint_ids, source_vertical_contact

class State:
  def __init__(self, env):
    n=env.num_envs; d=env.device
    self.flight=torch.zeros(n,dtype=torch.bool,device=d); self.landed=torch.zeros_like(self.flight)
    self.last_contact=torch.zeros((n,4),dtype=torch.bool,device=d); self.max_pitch_rate=torch.zeros(n,device=d)
    self.start_pos=torch.zeros((n,2),device=d)
    self.up_attempted=torch.zeros_like(self.flight); self.rot_attempted=torch.zeros_like(self.flight); self.steps=0; self.last_step=-1
  def update(self,env,command_name,sensor_name):
    if self.last_step==int(env.common_step_counter): return
    self.last_step=int(env.common_step_counter); self.steps+=1
    r: Entity=env.scene['robot']; c=env.command_manager.get_command(command_name); assert c is not None
    contact=source_vertical_contact(env.scene[sensor_name],1.); filt=contact|self.last_contact; self.last_contact.copy_(contact)
    self.flight |= ~filt.any(1)&(c[:,2]>0); self.landed |= filt.any(1)&self.flight
    self.max_pitch_rate=torch.maximum(self.max_pitch_rate,torch.abs(r.data.root_link_ang_vel_b[:,1]))
    # Source updates count through _reward_dof_vel and uses p=max(8-floor(count/1200),0)/10.
    p=max(8-self.steps//1200,0)/10.
    first=(~self.up_attempted)&(~self.landed)&(c[:,2]>0); ids=torch.nonzero(first&(torch.rand(env.num_envs,device=env.device)<p)).squeeze(1)
    if len(ids):
      v=r.data.root_link_vel_w[ids].clone(); v[:,2]+=torch.empty(len(ids),device=env.device).uniform_(2.,3.5); r.write_root_link_velocity_to_sim(v,env_ids=ids)
    self.up_attempted|=first
    second=self.up_attempted&self.flight&(~self.rot_attempted); ids=torch.nonzero(second&(torch.rand(env.num_envs,device=env.device)<p)).squeeze(1)
    if len(ids):
      v=r.data.root_link_vel_w[ids].clone(); v[:,4]+=torch.empty(len(ids),device=env.device).uniform_(2.,2.5); r.write_root_link_velocity_to_sim(v,env_ids=ids)
    self.rot_attempted|=second
  def reset(self,env,ids=None):
    if ids is None: ids=torch.arange(env.num_envs,device=env.device)
    self.flight[ids]=False; self.landed[ids]=False; self.last_contact[ids]=False; self.max_pitch_rate[ids]=0.; self.up_attempted[ids]=False; self.rot_attempted[ids]=False; self.start_pos[ids]=env.scene['robot'].data.root_link_pos_w[ids,:2]
def state(env):
  x=getattr(env,'_backflip_state',None)
  if x is None: x=State(env); setattr(env,'_backflip_state',x)
  return x
def euler(r): return torch.stack(euler_xyz_from_quat(r.data.root_link_quat_w),1)
class BeforeSetting:
  def __init__(self,cfg:RewardTermCfg,env): self.env=env; self.s=state(env)
  def __call__(self,env,command_name,sensor_name):
    self.s.update(env,command_name,sensor_name); r=env.scene['robot']; ids=joint_ids(r); c=env.command_manager.get_command(command_name)
    return torch.exp(-torch.abs(r.data.joint_pos[:,ids]-r.data.default_joint_pos[:,ids]).sum(1)/4)*(c[:,2]==0)
  def reset(self,env_ids=None): self.s.reset(self.env,env_ids)
def _u(env,n,s): x=state(env); x.update(env,n,s); return x
def line_z(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); r=env.scene['robot']; c=env.command_manager.get_command(command_name); return (r.data.root_link_vel_w[:,2]>0)*r.data.root_link_vel_w[:,2]*~x.landed*(c[:,2]==1)
def angle_y(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); y=env.scene['robot'].data.root_link_ang_vel_b[:,1]; c=env.command_manager.get_command(command_name); return torch.clamp(y*(y>0)*(c[:,2]==1)*(~x.flight)+3*y*(y>0)*(x.flight*~x.landed),max=20.)
def height_flight(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); return torch.exp(-torch.abs(env.scene['robot'].data.root_link_pos_w[:,2]-.6)*5)*x.flight*~x.landed*6
def height_stance(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); z=env.scene['robot'].data.root_link_pos_w[:,2]; return torch.exp(-torch.abs(z-.35)*5)*x.landed*(z>.2)
def orientation(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); return torch.exp(-torch.abs(euler(env.scene['robot'])).sum(1))*x.landed*(x.max_pitch_rate>7)
def orientation_before(env,command_name):
  c=env.command_manager.get_command(command_name); return torch.exp(-torch.abs(euler(env.scene['robot'])).sum(1))*(c[:,2]==0)
def land_pos(env,command_name,sensor_name):
  x=_u(env,command_name,sensor_name); return torch.exp(-torch.abs(x.start_pos-env.scene['robot'].data.root_link_pos_w[:,:2]).sum(1))*x.landed
def symmetric_joints(env):
  r=env.scene['robot']; q=r.data.joint_pos[:,joint_ids(r)].reshape(-1,4,3).clone(); q[:,1,0]*=-1; q[:,3,0]*=-1; return torch.abs(q[:,0]-q[:,1]).sum(1)+torch.abs(q[:,2]-q[:,3]).sum(1)
def dof_pos(env):
  r=env.scene['robot']; i=joint_ids(r); return torch.abs(r.data.joint_pos[:,i]-r.data.default_joint_pos[:,i]).sum(1)
def hip(env): return torch.abs(env.scene['robot'].data.joint_pos[:,joint_ids(env.scene['robot'])])[:,(0,3,6,9)].sum(1)
def ang_xy(env): return torch.abs(env.scene['robot'].data.root_link_ang_vel_b[:,(0,2)]).sum(1)
def torques(env): return torch.abs(env.scene['robot'].data.qfrc_actuator[:,joint_ids(env.scene['robot'])]).sum(1)
def action_rate(env): return torch.square(env.action_manager.action-env.action_manager.prev_action).sum(1)
def collision(env,sensor_name): return (torch.linalg.vector_norm(env.scene[sensor_name].data.force,dim=-1)>.1).sum(1)
def dof_pos_limits(env):
  r=env.scene['robot']; i=joint_ids(r); l=r.data.soft_joint_pos_limits; q=r.data.joint_pos[:,i]; return (-(q-l[:,i,0]).clamp(max=0)+(q-l[:,i,1]).clamp(min=0)).sum(1)
def dof_vel(env): return torch.square(env.scene['robot'].data.joint_vel[:,joint_ids(env.scene['robot'])]).sum(1)
def dof_vel_limits(env): return (torch.abs(env.scene['robot'].data.joint_vel[:,joint_ids(env.scene['robot'])])-30.).clamp(min=0).sum(1)
def feet_force(env,sensor_name,max_contact_force): return (torch.linalg.vector_norm(env.scene[sensor_name].data.force,dim=-1)-max_contact_force).clamp(min=0).sum(1)
def line_vel_stance(env): return torch.abs(env.scene['robot'].data.root_link_lin_vel_b).sum(1)
