from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
import optax
from flax.struct import PyTreeNode

from qdax.core.neuroevolution.buffers.buffer import Transition, QDTransition
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from src.models.base_utils import SurrogateModel, SurrogateModelState, SurrogateModelConfig
from src.models.base_models import (
    DynamicsModule, 
    make_dynamics_model_loss_fn,
    ProbDynamicsModule, 
    make_prob_dynamics_model_loss_fn,
)

from src.models.utils import ImprovedReplayBuffer as ReplayBuffer

@dataclass
class DynamicsModelConfig(SurrogateModelConfig):

    imagination_horizon: int = 200 # should correspond to episode length
    add_buffer_size: int = 200 # how many solutions to add to the buffer before ending imagination
    num_imagined_iterations: int = 10

    # model parameters
    surrogate_hidden_layer_sizes: Tuple[int, ...] = (128, 128)
    surrogate_ensemble_size: int = 1

    ts_inf: bool = False

    prob: bool = False
    learn_std: bool = False
    fixed_std: float = 0.01

    use_grad_clipping: bool = False
    grad_clip_value: float = 1.0

    num_batches_per_loss: int = 5

    # model training parameters
    surrogate_learning_rate: float = 0.001
    surrogate_batch_size: int = 512
    num_model_training_steps: int = 100
    surrogate_replay_buffer_size: int = 1000000
    surrogate_model_update_period: int = 10 # how often to update the model
    max_epochs_since_improvement: int = 5
    

class DynamicsModelState(SurrogateModelState):

    params: Params
    optimizer_state: optax.OptState
    replay_buffer: ReplayBuffer
    random_key: RNGKey
    loss: float



@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll_ts_inf(
    init_state: jnp.ndarray,
    policy_params: Params,
    model_idx: jnp.ndarray,
    model_params: Params,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [jnp.ndarray, Params, Params, jnp.ndarray, RNGKey],
        Tuple[
            jnp.ndarray,
            Params,
            Params,
            jnp.ndarray,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[jnp.ndarray, Transition]:
    """Generates and rolls out a imagined episode/evaluation according to the params, 
    returns the final state of the episode and the transitions of the episode.
    - used for ts_inf play step function
    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        surrogate_state: state of the surrogate model.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[jnp.ndarray, Params, Params, jnp.ndarray, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[jnp.ndarray, Params, Params, jnp.ndarray, RNGKey], Transition]:
        env_state, policy_params, model_params, model_idx, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, model_params, model_idx, random_key), transitions

    (state, _, _, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, model_params, model_idx, random_key),
        (),
        length=episode_length,
    )
    return state, transitions

@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll_ts1(
    init_state: jnp.ndarray,
    policy_params: Params,
    model_params: Params,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [jnp.ndarray, Params, Params, RNGKey],
        Tuple[
            jnp.ndarray,
            Params,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[jnp.ndarray, Transition]:
    """
    used for ts1 play step function
    """

    def _scan_play_step_fn(
        carry: Tuple[jnp.ndarray, Params, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[jnp.ndarray, Params, Params, RNGKey], Transition]:
        env_state, policy_params, model_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, model_params, random_key), transitions

    (state, _, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, model_params, random_key),
        (),
        length=episode_length,
    )
    return state, transitions



class DynamicsModel(SurrogateModel):
    '''
    Wraps DynamicsModule to provide a high-level interface for 
    (i) recursively rolling out the model in imagination, given a params/policies (getting one evaluation to get expected fitness and desc)
    (ii) training the model
    '''
    def __init__(
        self,
        config: SurrogateModelConfig,
        policy_network: nn.Module,
        state_size: int,
        action_size: int,
        play_reset_fn: Callable[[RNGKey], jnp.ndarray],
        reward_extractor_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bd_extractor_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ):
        self._config = config
        self._policy_network = policy_network
        self._state_size = state_size
        self._action_size = action_size
        self._play_reset_fn = play_reset_fn
        
        self.reward_extractor_fn = reward_extractor_fn
        self.bd_extractor_fn = bd_extractor_fn

        # Probabilistic or Deterministic Dynamics Model
        self._input_size = self._state_size + self._action_size
        if self._config.prob:
            
            if self._config.learn_std:
                self._output_size = self._state_size * 2
            else:
                self._output_size = self._state_size

            self.dynamics_model = ProbDynamicsModule(
                    input_size=self._input_size,
                    output_size=self._output_size,
                    hidden_layer_sizes=self._config.surrogate_hidden_layer_sizes,
                    input_mu=jnp.zeros(shape=(self._state_size+self._action_size,)),
                    input_std=jnp.ones(shape=(self._state_size+self._action_size,)),
                    output_mu=jnp.zeros(shape=(self._state_size,)),
                    output_std=jnp.ones(shape=(self._state_size,)),
                    learn_std=self._config.learn_std,
                    fixed_std=self._config.fixed_std,
            )

            # get loss function
            self._dynamics_model_loss_fn = make_prob_dynamics_model_loss_fn(dynamics_model_fn=self.dynamics_model.apply,
                                                                        learn_std=self._config.learn_std,
                                                                        fixed_std=self._config.fixed_std)

        else:
            
            self._output_size = self._state_size
            self.dynamics_model = DynamicsModule(
                input_size=self._input_size,
                output_size=self._output_size,
                hidden_layer_sizes=self._config.surrogate_hidden_layer_sizes,
                input_mu=jnp.zeros(shape=(self._state_size+self._action_size,)),
                input_std=jnp.ones(shape=(self._state_size+self._action_size,)),
                output_mu=jnp.zeros(shape=(self._state_size,)),
                output_std=jnp.ones(shape=(self._state_size,)),
            )

            # get loss function
            self._dynamics_model_loss_fn = make_dynamics_model_loss_fn(dynamics_model_fn=self.dynamics_model.apply)
        
        # init optimizer
        if self._config.use_grad_clipping:
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.grad_clip_value),
                optax.adam(learning_rate=self._config.surrogate_learning_rate),
            )
        else:
            self._optimizer = optax.adam(learning_rate=self._config.surrogate_learning_rate)


        if self._config.ts_inf:
            self.play_imagined_step_fn = self.play_imagined_step_ts_inf
            self.generate_unroll_fn = generate_unroll_ts_inf
        else:
            self.play_imagined_step_fn = self.play_imagined_step_ts1
            self.generate_unroll_fn = generate_unroll_ts1


    def init(self, random_key: RNGKey) -> Tuple[DynamicsModelState, RNGKey]:
        """
        Initializes the training state (model params and optimizer state) of the model
        """
        # init dynamics model params
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=self._config.surrogate_ensemble_size)
        fake_state = jnp.zeros(shape=(self._config.surrogate_ensemble_size, self._state_size,))
        fake_action = jnp.zeros(shape=(self._config.surrogate_ensemble_size, self._action_size,))
        init_params = jax.vmap(self.dynamics_model.init)(subkeys, fake_state, fake_action)

        # print("init_params dynamics model:", jax.tree_util.tree_map(lambda x: x.shape, init_params))

        # init optimizer
        optimizer_state = self._optimizer.init(init_params)

        # print("optimizer_state dynamics model:", jax.tree_util.tree_map(lambda x: x.shape, optimizer_state))
        
        # initialize replay buffer used to train dynamics model
        dummy_transition = QDTransition.init_dummy(
            observation_dim=self._state_size,
            action_dim=self._action_size,
            descriptor_dim=2, # CAREFUL - HARD CODED FOR NOW!
        )
        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.surrogate_replay_buffer_size, transition=dummy_transition
        )

        # initialize training state
        random_key, subkey = jax.random.split(random_key)
        training_state = DynamicsModelState(params=init_params, 
                                          optimizer_state=optimizer_state,
                                          replay_buffer=replay_buffer,
                                          random_key=subkey,
                                          loss=jnp.inf)

        return training_state, random_key

    def play_imagined_step_ts_inf(
        self,
        state: jnp.ndarray,
        policy_params: Params,
        model_params: Params,
        model_idx: jnp.ndarray,
        random_key: RNGKey,
        ) -> Tuple[jnp.ndarray, Params, Params, RNGKey, Transition]:
        """
        Play an imagined step using the dynamics model and return the updated state and the transition.
        - ts inf model_idx is the index of the model in the ensemble to use - trajectory shooting according to model
        - particle bootstraps never changing during a trial
        """
        random_key, prob_key = jax.random.split(random_key, num=2)

        actions = self._policy_network.apply(policy_params, state)
        
        # expand dims to match the ensemble size
        actions_e = jnp.repeat(jnp.expand_dims(actions, axis=0), self._config.surrogate_ensemble_size, axis=0)
        state_e = jnp.repeat(jnp.expand_dims(state, axis=0), self._config.surrogate_ensemble_size, axis=0)

        # predict next state with probablistic ensemble model - same API as deterministic API (takes in key) but doenst do anything with it
        prob_keys = jax.random.split(prob_key, num=self._config.surrogate_ensemble_size)
        delta_next_state_e = jax.vmap(self.dynamics_model.get_pred)(model_params, state_e, actions_e, prob_keys)
        # print("Delta next shape ensemble: ", delta_next_state_e.shape)
        
        delta_next_state = delta_next_state_e[model_idx]
        # print("Delta next shape selected: ", delta_next_state.shape)

        next_state = state + delta_next_state
        rwd = self.reward_extractor_fn(next_state, state, actions)

        transition = QDTransition(
            obs=state,
            next_obs=next_state,
            rewards=rwd, # CAREFUL WARNING: HARD CODED FOR NOW
            dones=0, # CAREFUL WARNING: HARD CODED FOR NOW
            actions=actions,
            truncations=0, # CAREFUL WARNING: HARD CODED FOR NOW
            state_desc=state[:2], # CAREFUL WARNING: HARD CODED FOR NOW
            next_state_desc=next_state[:2], # CAREFUL WARNING: HARD CODED FOR NOW
        )

        return next_state, policy_params, model_params, model_idx, random_key, transition


    def play_imagined_step_ts1(
        self,
        state: jnp.ndarray,
        policy_params: Params,
        model_params: Params,
        random_key: RNGKey,
        ) -> Tuple[jnp.ndarray, Params, Params, RNGKey, Transition]:
        """
        Play an imagined step and return the updated state and the transition - function implement for one genotype (can vmaap to get batch)
        - select the next state randomnly from any of the models in the ensmeble
        - particles uniformly re-sampling a bootstrap per time step
        """
        random_key, prob_key, ensemble_key = jax.random.split(random_key, num=3)

        actions = self._policy_network.apply(policy_params, state)
        
        # expand dims to match the ensemble size
        actions_e = jnp.repeat(jnp.expand_dims(actions, axis=0), self._config.surrogate_ensemble_size, axis=0)
        state_e = jnp.repeat(jnp.expand_dims(state, axis=0), self._config.surrogate_ensemble_size, axis=0)

        # predict next state with probablistic ensemble model - same API as deterministic API (takes in key) but doenst do anything with it
        prob_keys = jax.random.split(prob_key, num=self._config.surrogate_ensemble_size)
        delta_next_state_e = jax.vmap(self.dynamics_model.get_pred)(model_params, state_e, actions_e, prob_keys)
        
        # sample one of the models uniformly for the next state
        model_idx = jax.random.randint(ensemble_key, shape=(), minval=0, maxval=self._config.surrogate_ensemble_size)
        delta_next_state = delta_next_state_e[model_idx]

        next_state = state + delta_next_state
        rwd = self.reward_extractor_fn(next_state, state, actions)

        transition = QDTransition(
            obs=state,
            next_obs=next_state,
            rewards=rwd, # CAREFUL WARNING: HARD CODED FOR NOW
            dones=0, # CAREFUL WARNING: HARD CODED FOR NOW
            actions=actions,
            truncations=0, # CAREFUL WARNING: HARD CODED FOR NOW
            state_desc=state[:2], # CAREFUL WARNING: HARD CODED FOR NOW
            next_state_desc=next_state[:2], # CAREFUL WARNING: HARD CODED FOR NOW
        )

        return next_state, policy_params, model_params, random_key, transition


    
    @partial(jax.jit, static_argnames=("self",))
    def scoring_function(
        self, 
        policy_params: Genotype, 
        random_key: RNGKey,
        model_params: Params,
        ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        Returns the expected fitness and descriptors of the genotypes provided as computed by the surrogate model.
        """

        # automatically gives you a correct batch size of init states based on the policy params batch size
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, jax.tree_util.tree_leaves(policy_params)[0].shape[0])
        reset_fn = jax.vmap(self._play_reset_fn)
        init_states = reset_fn(keys)
        
        # sample one of the models uniformly for each individual rollout - batch of models needed
        random_key, subkey = jax.random.split(random_key)
        batch_size = jax.tree_util.tree_leaves(policy_params)[0].shape[0]
        model_idx = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=self._config.surrogate_ensemble_size)

        # Perform rollouts with each policy
        random_key, subkey = jax.random.split(random_key)
        unroll_fn = partial(
            self.generate_unroll_fn,
            model_params=model_params,
            random_key=subkey,
            episode_length=self._config.imagination_horizon,
            play_step_fn=self.play_imagined_step_fn,
        )

        _final_state, data = jax.vmap(unroll_fn)(init_states, policy_params, model_idx)

        # print("Shape of imagined rewards: ", data.rewards.shape) # (B, T)
        # print("Shape of imagined obs trajectory: ",data.obs.shape) # (B, T, D)
        fitnesses = jnp.sum(data.rewards, axis=1)
        descriptors = self.bd_extractor_fn(data.obs, data.actions)

        return (
            fitnesses,
            descriptors,
            {
                "transitions": data,
            },
            random_key,
        )

    def update(
        self,
        surrogate_state: DynamicsModelState, 
        extra_scores: ExtraScores,
    ) -> DynamicsModelState:
        '''update the entire dynamics model state by adding transitions to the replay buffer'''
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = surrogate_state.replay_buffer.insert(transitions)
        replay_buffer = replay_buffer.clean_up_buffer()
        surrogate_state = surrogate_state.replace(replay_buffer=replay_buffer)

        return surrogate_state

    ################################
    ### TRAINING MODEL FUNCTIONS ###
    def _train_model_body_fun(self, state):
        """
        Body function of the while loop that trains the dynamics model.
        """
        (
            train_data,
            test_data,
            surrogate_state,
            random_key,
            best_test_loss,
            epochs_since_improvement,
            training_steps,
        ) = state

        replay_buffer = surrogate_state.replay_buffer
        model_params = surrogate_state.params
        optimizer_state = surrogate_state.optimizer_state

        _loss_fn = partial(self._dynamics_model_loss_fn, output_mu=self.dynamics_model.output_mu, output_std=self.dynamics_model.output_std)
        n_train = train_data.shape[0]
        num_batches = self._config.num_batches_per_loss #n_train // self._config.surrogate_batch_size
        for it in range(num_batches):
            # Sample a batch of transitions in the train set - batchsize*ensemble_size
            samples, random_key = replay_buffer.sample_data(
                random_key, train_data, sample_size=self._config.surrogate_batch_size*self._config.surrogate_ensemble_size
            )
            
            # Reshape so we can train bootstrapped ensembel models
            # print("Samples obs shape: ", samples.obs.shape)
            samples = jax.tree_util.tree_map(
                lambda x: x.reshape((self._config.surrogate_ensemble_size, self._config.surrogate_batch_size, -1)),
                samples
            )
            # print("Reshaped samples obs shape: ", samples.obs.shape)
            
            # Compute the loss and update the model
            loss, gradient = jax.vmap(jax.value_and_grad(_loss_fn))(
                model_params,
                samples,
            )
            # print("Loss shape: ", loss.shape)
            # print("Gradient shape: ", jax.tree_util.tree_map(lambda x: x.shape, gradient))
            
            (model_updates, optimizer_state,) = self._optimizer.update(
                gradient, optimizer_state
            )
            # print("Model updates shape: ", jax.tree_util.tree_map(lambda x: x.shape, model_updates))
            # print("Optimizer state shape: ", jax.tree_util.tree_map(lambda x: x.shape, optimizer_state))

            model_params = jax.vmap(optax.apply_updates)(model_params, model_updates)
            # print("Model params shape: ", jax.tree_util.tree_map(lambda x: x.shape, model_params))
            # print(f"Training Loss {it}/{num_batches}: ", loss)

        training_steps = training_steps + num_batches

        # Compute the test loss
        test_samples, random_key = replay_buffer.sample_data(
            random_key, test_data, sample_size=self._config.surrogate_batch_size
        )
        # print("Test samples obs shape: ", jax.tree_util.tree_map(lambda x: x.shape, test_samples))
        test_samples = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self._config.surrogate_ensemble_size, axis=0),
            test_samples
        )
        # print("Reshaped test samples obs shape: ", jax.tree_util.tree_map(lambda x: x.shape, test_samples))
        test_loss, _ = jax.lax.stop_gradient(
            jax.vmap(
                jax.value_and_grad(_loss_fn))(
                model_params,
                test_samples,
                )
        )

        test_loss = jnp.mean(test_loss)
        # print(f" Holdout Loss {training_steps}: ", test_loss)

        def cond_true(best_test_loss, test_loss, epochs_since_improvement):
            best_test_loss = test_loss
            epochs_since_improvement = 0
            return best_test_loss, epochs_since_improvement
        def cond_false(best_test_loss, test_loss, epochs_since_improvement):
            epochs_since_improvement += 1
            return best_test_loss, epochs_since_improvement
        
        # less than 1% improvement in test loss - early stopping
        # print("best_test_loss: ", best_test_loss)
        # print("test_loss: ", test_loss)
        # print("Condition", (best_test_loss - test_loss) / best_test_loss)
        best_test_loss, epochs_since_improvement = jax.lax.cond(
            (best_test_loss - test_loss)/abs(best_test_loss) > 0.01,
            cond_true,
            cond_false,
            best_test_loss, test_loss, epochs_since_improvement
        )

        surrogate_state = surrogate_state.replace(
            params=model_params,
            optimizer_state=optimizer_state,
            loss=jnp.mean(loss),
        )   
        return (
            train_data,
            test_data,
            surrogate_state,
            random_key,
            best_test_loss,
            epochs_since_improvement,
            training_steps,
        )

    def _train_model_cond_fun(self, state):
        """
        Termination condition of the training of the model
        - test_loss stagnates for a certain number of epochs or
        - the number of gradient steps exceeds the maximum number of training steps
        """
        (   
            train_data,
            test_data,
            surrogate_state,
            random_key,
            best_test_loss,
            epochs_since_improvement,
            training_steps,
        ) = state
        
        # print("Epochs since improvement: ", epochs_since_improvement)
        # print("Training steps: ", training_steps)
        return jnp.logical_and(
            epochs_since_improvement < self._config.max_epochs_since_improvement,
            training_steps < self._config.num_model_training_steps
        )

    #@partial(jax.jit, static_argnames=("self",))
    def _train_model_simple(self, surrogate_state, random_key):
        """
        Train the dynamics model using the entire dataset for a fixed number of specified training steps
        """

        replay_buffer = surrogate_state.replay_buffer
        model_params = surrogate_state.params
        optimizer_state = surrogate_state.optimizer_state

        _loss_fn = partial(self._dynamics_model_loss_fn, output_mu=self.dynamics_model.output_mu, output_std=self.dynamics_model.output_std)

        for _ in range(self._config.num_model_training_steps):
            
            # Sample a batch of transitions in the train set - batchsize*ensemble_size
            samples, random_key = replay_buffer.sample(
                random_key, sample_size=self._config.surrogate_batch_size*self._config.surrogate_ensemble_size
            )
            
            # Reshape so we can train bootstrapped ensembel models
            # print("Samples obs shape: ", samples.obs.shape)
            samples = jax.tree_util.tree_map(
                lambda x: x.reshape((self._config.surrogate_ensemble_size, self._config.surrogate_batch_size, -1)),
                samples
            )
            # print("Reshaped samples obs shape: ", samples.obs.shape)
            
            # Compute the loss and update the model
            loss, gradient = jax.vmap(jax.value_and_grad(_loss_fn))(
                model_params,
                samples,
            )
            # print("Loss shape: ", loss.shape)
            # print("Gradient shape: ", jax.tree_util.tree_map(lambda x: x.shape, gradient))
            
            #@jax.jit
            def _update_params(params_input: Tuple) -> Tuple:
                gradient, optimizer_state, model_params = params_input
                (model_updates, optimizer_state,) = self._optimizer.update(
                    gradient, optimizer_state
                )
                # print("Model updates shape: ", jax.tree_util.tree_map(lambda x: x.shape, model_updates))
                # print("Optimizer state shape: ", jax.tree_util.tree_map(lambda x: x.shape, optimizer_state))

                model_params = jax.vmap(optax.apply_updates)(model_params, model_updates)
                # print("Model params shape: ", jax.tree_util.tree_map(lambda x: x.shape, model_params))
                # print(f"Training Loss {it}/{num_batches}: ", loss)
                return (model_params, optimizer_state)

            (model_params, optimizer_state) = jax.lax.cond(
                jnp.any(jnp.isnan(loss)),
                lambda x: (x[2], x[1]),
                _update_params,
                (gradient, optimizer_state, model_params),
            )

        # update surrogate state
        surrogate_state = surrogate_state.replace(
            params=model_params,
            optimizer_state=optimizer_state,
            loss=jnp.mean(loss),
        )  

        return surrogate_state, random_key



    def train_model(
        self,
        surrogate_state: DynamicsModelState
    ) -> DynamicsModelState:
        """
        Trains the deep dynamics model
        - 2 strategies (1) simple training for fixed number of epochs (2) training with early stopping (based on test loss)
        """

        random_key = surrogate_state.random_key
        replay_buffer = surrogate_state.replay_buffer
        
        all_transitions = replay_buffer.get_all_transitions()
        assert not jnp.any(jnp.isnan(all_transitions.obs)), "NaNs in observations"
        assert not jnp.any(jnp.isnan(all_transitions.actions)), "NaNs in actions"
        self.dynamics_model.fit_input_stats(all_transitions.obs, all_transitions.actions)
        self.dynamics_model.fit_output_stats(all_transitions.obs, all_transitions.next_obs)

        # split data into train and test sets
        train_data, test_data, random_key = replay_buffer.train_test_split(random_key)

        # train model for n steps or until test loss stagnates
        with jax.disable_jit():
            (_, _, new_surrogate_state, random_key, _, _, _) = jax.lax.while_loop(
                self._train_model_cond_fun, 
                self._train_model_body_fun,
                (train_data, test_data, surrogate_state, random_key, 10e6, 0, 0)
            )

        # (new_surrogate_state, random_key) = self._train_model_simple(surrogate_state, random_key)

        # Create new training state
        new_training_state = DynamicsModelState(
            params=new_surrogate_state.params,
            optimizer_state=new_surrogate_state.optimizer_state,
            replay_buffer=replay_buffer,
            random_key=random_key,
            loss=new_surrogate_state.loss
        )

        return new_training_state


    
  
