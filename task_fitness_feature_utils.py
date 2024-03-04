import jax.numpy as jnp
from brax.math import quat_to_euler

####################################################
##### COMMON FUNCTIONS USED TO COMPUTE REWARDS #####
####################################################

def forward_vel_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    
    vel_rwd = obs.at[0].get() - prev_obs.at[0].get() / 0.05 # dt is 0.05 HARDCODED FOR NOW

    return vel_rwd

def action_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises large actions
    """
    rwd = -0.5*jnp.sum(jnp.square(actions))
    return rwd

def forward_vel_reward_humanoid(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
        
    vel_rwd = obs.at[0].get() - prev_obs.at[0].get() / 0.015 # dt is 0.015 HARDCODED FOR NOW

    return vel_rwd

def action_reward_humanoid(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises large actions
    """
    rwd = -0.1*jnp.sum(jnp.square(actions)) # 0.1 coeff is HARDCODED FOR NOW
    return rwd


def _angle_dist(a, b):
    theta = b - a

    theta = jnp.where(theta < -jnp.pi, theta + 2 * jnp.pi, theta)
    theta = jnp.where(theta > jnp.pi, theta - 2 * jnp.pi, theta)

    theta = jnp.where(theta < -jnp.pi, theta + 2 * jnp.pi, theta)
    theta = jnp.where(theta > jnp.pi, theta - 2 * jnp.pi, theta)

    return theta

def angle_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    angle error reward for custom ant omni based on x,y location there is a desired yaw angle
    """
    com_rot = quat_to_euler(obs.at[3:7].get())
    z_rot = com_rot.at[2].get()

    ang_x, ang_y = obs.at[0].get(), obs.at[1].get()

    B = jnp.sqrt((ang_x / 2.0) * (ang_x / 2.0) + (ang_y / 2.0) * (ang_y / 2.0))
    alpha = jnp.arctan2(ang_y, ang_x)
    A = B / jnp.cos(alpha)
    beta = jnp.arctan2(ang_y, ang_x - A)

    beta = jnp.where(ang_x < 0,beta-jnp.pi,beta)

    beta = jnp.where(beta < -jnp.pi,beta + 2 * jnp.pi, beta)
    beta = jnp.where(beta > jnp.pi,beta - 2 * jnp.pi, beta)

    beta = jnp.where(beta < -jnp.pi,beta + 2 * jnp.pi, beta)
    beta = jnp.where(beta > jnp.pi,beta - 2 * jnp.pi, beta)

    beta = jnp.where(beta < -jnp.pi,beta + 2 * jnp.pi, beta)
    beta = jnp.where(beta > jnp.pi,beta - 2 * jnp.pi, beta)
            
    angle_diff = jnp.abs(_angle_dist(beta, z_rot))
    rwd = -angle_diff

    return rwd

#####################################################################################
################## DYNAMICS MODEL REWARDS FOR RESPECTIVE ENVS #######################
#####################################################################################

def antmaze_target_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
        
    xy_target = jnp.array([35.0, 0.0]) # CAREFUL HARDCODED FOR NOW
    xy_pos = obs.at[0:2].get()
    rwd = -jnp.linalg.norm(xy_pos - xy_target)
    return rwd

def pointmaze_target_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
        
    xy_target = jnp.array([-0.5, 0.8]) # CAREFUL HARDCODED FOR NOW
    xy_pos = obs.at[0:2].get()
    rwd = -jnp.linalg.norm(xy_pos - xy_target)
    return rwd

def anttrap_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises large actions
    """
    vel_rwd = forward_vel_reward(obs, prev_obs, actions)
    action_rwd = action_reward(obs, prev_obs, actions)

    return vel_rwd + action_rwd

def ant_xy_forward_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises large actions
    """
    vel_rwd = forward_vel_reward(obs, prev_obs, actions)
    action_rwd = action_reward(obs, prev_obs, actions)

    return vel_rwd + action_rwd

def ant_omni_action_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises large actions
    """
    action_rwd = action_reward(obs, prev_obs, actions)
    return action_rwd

def ant_omni_angle_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    reward for custom ant omni which penalises angle error
    """
    angle_rwd = angle_reward(obs, prev_obs, actions)
    return angle_rwd


fitness_extractor_imagination = {
    "pointmaze": pointmaze_target_reward,
    "anttrap": anttrap_reward,
    "antmaze": antmaze_target_reward,
    "ant_omni_action": ant_omni_action_reward,
    "ant_omni_angle": ant_omni_angle_reward,
    "ant_xy_forward": ant_xy_forward_reward,
}

###################################
########## BD EXTRACTORS ##########
###################################
def get_final_xy_position(obs: jnp.ndarray, actions:jnp.ndarray) -> float:
    """
    B, T, D = obs.shape
    """
    xy_pos = obs.at[:, -1, 0:2].get()
    
    return xy_pos


bd_extractor_imagination = {    
    "pointmaze": get_final_xy_position,
    "anttrap": get_final_xy_position,
    "antmaze": get_final_xy_position,
    "ant_omni_action": get_final_xy_position,
    "ant_omni_angle": get_final_xy_position,
    "ant_xy_forward": get_final_xy_position,   
}
