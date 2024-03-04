from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
import optax

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import EnvState, Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from src.models.utils import Datapoint

from brax.training.distribution import NormalDistribution


## utility functions for probablitic models (learning distributions)
## scale is std
    
## when learning mean only
def create_fixed_dist(mean, std=0.01):
    return NormalDistribution(loc=mean, scale=std)

def sample_fixed_dist(mean, std, key):
    return create_fixed_dist(mean, std).sample(seed=key)

def log_prob_fixed_dist(mean, std, value):
    return create_fixed_dist(mean, std).log_prob(value)

## when learning both mean and log var
def create_dist(parameters, min_logstd=-4.0, max_logstd=1.0):
    loc, logstd = jnp.split(parameters, 2, axis=-1)

    # Variance clamping to prevent poor numerical predictions
    logstd = max_logstd - jax.nn.softplus(max_logstd - logstd)
    logstd = min_logstd + jax.nn.softplus(logstd - min_logstd)
    std = jnp.exp(logstd)

    return NormalDistribution(loc=loc, scale=std)

def sample_dist(parameters, key):
    return create_dist(parameters).sample(seed=key)

def log_prob_dist(parameters, value):
    return create_dist(parameters).log_prob(value)



class DynamicsModule(nn.Module):
    """
    Dynamics Module
    - defined and initiliazed the dynamics model
    - implements the forward pass of the dynamics model (takes in s,a and returns s')
    """

    input_size: int # state_size + action_size
    output_size: int # state_size
    hidden_layer_sizes: Tuple[int, ...]

    # normalization params
    input_mu: jnp.ndarray # jnp.zeros(shape=(29+8,))
    input_std: jnp.ndarray # jnp.ones(shape=(29+8,))
    output_mu: jnp.ndarray # jnp.zeros(shape=(29,))
    output_std: jnp.ndarray # jnp.ones(shape=(29,))

    @nn.compact
    def __call__(self, state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([state, actions], axis=-1)
        hidden = self.normalize_inputs(hidden)

        q = MLP(
                layer_sizes=self.hidden_layer_sizes + (self.output_size,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_normal(),
                )(hidden)
    
        return q

    def get_pred(self, params, state: jnp.ndarray, actions: jnp.ndarray, random_key: RNGKey) -> jnp.ndarray:
        delta_state_norm = self.apply(params, state, actions)
        delta_state = self.denormalize_outputs(delta_state_norm)
        return delta_state


    def fit_input_stats(self, states, actions):
        '''
        get normalization params of input data
        '''

        data = jnp.concatenate([states, actions], axis=-1)
        mean = jnp.mean(data, axis=0, keepdims=True).squeeze(axis=0)
        std = jnp.std(data, axis=0, keepdims=True).squeeze(axis=0)
        std = std.at[std != std].set(1.0)
        std = std.at[std < 1e-12].set(1.0)
 
        self.input_mu = mean
        self.input_std = std

    def fit_output_stats(self, obs, next_obs):
        '''get normalization params of output data'''
        # learn the delta next state as done in PETS (Chua et al. 2018)
        data = next_obs - obs 

        mean = jnp.mean(data, axis=0, keepdims=True).squeeze(axis=0)
        std = jnp.std(data, axis=0, keepdims=True).squeeze(axis=0)
        std = std.at[std != std].set(1.0)
        std = std.at[std < 1e-12].set(1.0)

        self.output_mu = mean
        self.output_std = std

    def normalize_inputs(self, data):
        '''normalize input data'''
        return (data - self.input_mu) / (self.input_std + 1e-6)

    def normalize_outputs(self, data):
        '''normalize output data'''
        return (data - self.output_mu) / (self.output_std + 1e-6)

    def denormalize_outputs(self, data):
        '''de-normalize outputput data'''
        return data*self.output_std + self.output_mu


def make_dynamics_model_loss_fn(
    dynamics_model_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[Params, Transition], jnp.ndarray]:
    """Creates the loss used to train the dynamics model.
    Args:
        dynamics_model_fn: the apply function of the dynamics model
    Returns:
        the loss of the dynamics model
    """

    @jax.jit
    def _dynamics_model_loss_fn(
        model_params: Params,
        transitions: Transition,
        output_mu: jnp.ndarray,
        output_std: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        We want model to learn the normalized s'
        Given a transition with obs, action and next_obs, compute the loss of the dynamics model prediction
        make prediciton with dynamics model of next_obs from obs and action
        get loss between prediction and next_obs
        """

        # get prediction from model
        pred_delta_next_state_norm = dynamics_model_fn(model_params, transitions.obs, transitions.actions)
        
        # compute targets (learn the delta next state)
        target_delta_next_state = transitions.next_obs - transitions.obs 
        target_delta_next_state_norm = (target_delta_next_state - output_mu)/(output_std + 1e-6)

        # loss 
        loss = jnp.mean(jnp.sum(jnp.square(pred_delta_next_state_norm - target_delta_next_state_norm), axis=-1), axis=-1)
        return loss

    return _dynamics_model_loss_fn


class ProbDynamicsModule(DynamicsModule):
    """
    Probablistic Dynamics Model
    - input and output size of the model is potentially different from deterministic model if learning variance of distrubution
    """

    learn_std: bool = False
    fixed_std: float = 0.01

    def get_pred(self, params, state: jnp.ndarray, actions: jnp.ndarray, random_key: RNGKey) -> jnp.ndarray:
        '''
        Model only gives the mean of the distributions - we then need to construct the distribution and sample from it
        '''
        delta_state_norm = self.apply(params, state, actions)
        if self.learn_std:
            delta_state_norm = sample_dist(delta_state_norm, random_key)
        else: 
            delta_state_norm = sample_fixed_dist(delta_state_norm, self.fixed_std, random_key)

        delta_state = self.denormalize_outputs(delta_state_norm)
        return delta_state 


def make_prob_dynamics_model_loss_fn(
    dynamics_model_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    learn_std: bool = False,
    fixed_std: float = 0.01,
) -> Callable[[Params, Transition], jnp.ndarray]:
    """Creates the loss used to train the dynamics model.
    Args:
        dynamics_model_fn: the apply function of the dynamics model - make sure we put the log prob one
    Returns:
        the loss of the dynamics model
    """
    if learn_std:
        @jax.jit
        def _dynamics_model_loss_fn(
            model_params: Params,
            transitions: Transition,
            output_mu: jnp.ndarray,
            output_std: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Given a transition with obs, action and next_obs, 
            compute the negative log likelihood loss of the dynamics model prediction
            """

            # get prediction from model
            pred_delta_next_state_norm = dynamics_model_fn(model_params, transitions.obs, transitions.actions)

            # compute targets
            target_delta_next_state = transitions.next_obs - transitions.obs 
            target_delta_next_state_norm = (target_delta_next_state - output_mu)/(output_std + 1e-6)

            # negative log likelihood loss
            loss = -jnp.mean(jnp.sum(log_prob_dist(pred_delta_next_state_norm, target_delta_next_state_norm), axis=-1), axis=-1)

            return loss
    else: 
        @jax.jit
        def _dynamics_model_loss_fn(
            model_params: Params,
            transitions: Transition,
            output_mu: jnp.ndarray,
            output_std: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Given a transition with obs, action and next_obs, 
            compute the negative log likelihood loss of the dynamics model prediction
            """
            # get prediction from model
            pred_delta_next_state_norm = dynamics_model_fn(model_params, transitions.obs, transitions.actions)
            
            # compute targets
            target_delta_next_state = transitions.next_obs - transitions.obs 
            target_delta_next_state_norm = (target_delta_next_state - output_mu)/(output_std + 1e-6)

            # negative log likelihood loss
            loss = -jnp.mean(jnp.sum(log_prob_fixed_dist(pred_delta_next_state_norm, fixed_std, target_delta_next_state_norm), axis=-1), axis=-1)
            
            return loss

    return _dynamics_model_loss_fn



class DirectModule(nn.Module):
    """Direct Module - Deep Neural Network"""

    input_size: int # genotype size dimensions
    output_size: int # fitness + bd dimensions
    hidden_layer_sizes: Tuple[int, ...]

    # normalization params
    input_mu: jnp.ndarray
    input_std: jnp.ndarray
    output_mu: jnp.ndarray
    output_std: jnp.ndarray

    @nn.compact
    def __call__(self, genotype: jnp.ndarray) -> jnp.ndarray:
        
        hidden = self.normalize_inputs(genotype)

        q = MLP(
                layer_sizes=self.hidden_layer_sizes + (self.output_size,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_normal(),
                )(hidden)
    
        return q

    def get_pred(self, params, genotype: jnp.ndarray, random_key: RNGKey) -> jnp.ndarray:
        fit_bd_norm = self.apply(params, genotype)
        fit_bd = self.denormalize_outputs(fit_bd_norm)
        return fit_bd


    def fit_input_stats(self, genotypes):
        '''
        get normalization params of input data
        '''

        data = genotypes
        mean = jnp.mean(data, axis=0, keepdims=True).squeeze(axis=0)
        std = jnp.std(data, axis=0, keepdims=True).squeeze(axis=0)
        std = std.at[std != std].set(1.0)
        std = std.at[std < 1e-12].set(1.0)

        self.input_mu = mean
        self.input_std = std

    def fit_output_stats(self, fitness, bd):
        '''
        get normalization params of input data
        '''
        data = jnp.concatenate([fitness, bd], axis=-1)
        #print("Delta next state: ", data.shape)
        mean = jnp.mean(data, axis=0, keepdims=True).squeeze(axis=0)
        std = jnp.std(data, axis=0, keepdims=True).squeeze(axis=0)
        std = std.at[std != std].set(1.0)
        std = std.at[std < 1e-12].set(1.0)

        self.output_mu = mean
        self.output_std = std

    def normalize_inputs(self, data):
        '''
        normalize input data
        '''
        return (data - self.input_mu) / (self.input_std + 1e-6)

    def normalize_outputs(self, data):
        '''
        normalize output data
        '''
        return (data - self.output_mu) / (self.output_std + 1e-6)

    def denormalize_outputs(self, data):
        '''
        normalize input data
        '''
        return data*self.output_std + self.output_mu


def make_direct_model_loss_fn(
    dynamics_model_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[Params, Datapoint], jnp.ndarray]:
    """Creates the loss used to train the direct model.
    Args:
        direct_model_fn: the apply function of the direct model

    Returns:
        the loss of the direct model
    """

    @jax.jit
    def _direct_model_loss_fn(
        model_params: Params,
        data: Datapoint,
        output_mu: jnp.ndarray,
        output_std: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Given a datapoint class with input genotypes, fitness and bd, compute the loss of the direct model prediction
        """
        # get predictions
        pred_fit_bd_norm = dynamics_model_fn(model_params, data.genotype)
      
        # compute target
        target_fit_bd = jnp.concatenate([jnp.expand_dims(data.fitness, axis=-1), data.desc], axis=-1)
        # print("Train model target delta next state: ", target_delta_next_state.shape)
        target_fit_bd_norm = (target_fit_bd - output_mu)/(output_std + 1e-6)
        loss = jnp.mean(jnp.sum(jnp.square(pred_fit_bd_norm - target_fit_bd_norm), axis=-1), axis=-1)
        # print("Loss shape: ",loss.shape)
        return loss

    return _direct_model_loss_fn


class ProbDirectModule(DirectModule):
    """
    Probablistic Direct Model
    """

    learn_std: bool = False
    fixed_std: float = 0.01

    def get_pred(self, params, genotype: jnp.ndarray, random_key: RNGKey) -> jnp.ndarray:
        '''
        Model only gives the mean of the distributions - we then need to construct the distribution and sample from it
        '''
        fit_bd_norm = self.apply(params, genotype)
        if self.learn_std:
            fit_bd_norm = sample_dist(fit_bd_norm, random_key) # in this case, fit_bd_norm will vs 2x the size of fit+bd dim
        else: 
            fit_bd_norm = sample_fixed_dist(fit_bd_norm, self.fixed_std, random_key)

        fit_bd = self.denormalize_outputs(fit_bd_norm)
        return fit_bd
    
    # write functions to get the mean and std of the predictions only
    def get_pred_dist(self, params, genotype: jnp.ndarray, random_key: RNGKey):
        '''
        To get the predictied mean and std of the distribution
        '''
        fit_bd_norm = self.apply(params, genotype)

        if self.learn_std:
            dist = create_dist(fit_bd_norm)
        else:
            dist = create_fixed_dist(fit_bd_norm, self.fixed_std)

        sample = dist.sample(seed=random_key)

        # return mean and std
        return dist.loc, dist.scale, sample


def make_prob_direct_model_loss_fn(
    direct_model_fn: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    learn_std: bool = False,
    fixed_std: float = 0.01,
) -> Callable[[Params, Datapoint], jnp.ndarray]:
    """Creates the loss used to train the dynamics model.

    Args:
        direct_model_fn: the apply function of the direct model - make sure we put the log prob one
    Returns:
        the loss of the direct model
    """
    if learn_std:
        @jax.jit
        def _direct_model_loss_fn(
            model_params: Params,
            data: Datapoint,
            output_mu: jnp.ndarray,
            output_std: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Given a transition with obs, action and next_obs, 
            compute the negative log likelihood loss of the dynamics model prediction
            """

            target_fit_bd = jnp.concatenate([jnp.expand_dims(data.fitness, axis=-1), data.desc], axis=-1)
            target_fit_bd_norm = (target_fit_bd - output_mu)/(output_std + 1e-6)
           
            pred_fit_bd_norm = direct_model_fn(model_params, data.genotype)
         
            loss = -jnp.mean(jnp.sum(log_prob_dist(pred_fit_bd_norm, target_fit_bd_norm), axis=-1), axis=-1)
            
            # print("Loss shape: ",loss.shape)
            return loss
    else: 
        @jax.jit
        def _direct_model_loss_fn(
            model_params: Params,
            data: Datapoint,
            output_mu: jnp.ndarray,
            output_std: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Given a transition with obs, action and next_obs, 
            compute the negative log likelihood loss of the dynamics model prediction
            """

            target_fit_bd = jnp.concatenate([data.fitness, data.desc], axis=-1)
            target_fit_bd_norm = (target_fit_bd - output_mu)/(output_std + 1e-6)
           
            pred_fit_bd_norm = direct_model_fn(model_params, data.genotype)
          
            loss = -jnp.mean(jnp.sum(log_prob_fixed_dist(pred_fit_bd_norm, fixed_std, target_fit_bd_norm), axis=-1), axis=-1)
            
            # print("Loss shape: ",loss.shape)
            return loss

    return _direct_model_loss_fn
