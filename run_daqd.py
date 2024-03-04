import os
from dataclasses import dataclass
import logging
import functools
from typing import Dict, Tuple
import pickle
import matplotlib.pyplot as plt
import time

import jax
import jax.numpy as jnp

import flax

from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as reset_based_scoring_function


from src.mbqd import ModelBasedMAPElites
from src.models.dynamics_model import DynamicsModel as DynamicsModelEnsemble
from src.models.dynamics_model import DynamicsModelConfig
from task_fitness_feature_utils import bd_extractor_imagination, fitness_extractor_imagination


from qdax.utils.plotting import plot_map_elites_results
from qdax.utils.plotting import plot_2d_map_elites_repertoire

import hydra


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""
    alg_name: str
    # Env config
    seed: int
    env_name: str
    episode_length: int
    policy_hidden_layer_sizes: Tuple[int, ...]
    # ME config
    num_evaluations: int
    batch_size: int
    single_init_state: bool
    discard_dead: bool
    # Grid config
    grid_shape: Tuple[int, ...]
    # Emitter config
    iso_sigma: float
    line_sigma: float
    crossover_percentage: float
    # others
    log_period: int # only for timings and metrics
    store_repertoire: bool 
    store_repertoire_log_period: int
    plot_grid: bool
    plot_grid_log_period: int

    # surrogate model parameters
    num_imagined_iterations: int
    add_buffer_size: int
    # model parameters
    surrogate_hidden_layer_sizes: Tuple[int, ...]
    surrogate_ensemble_size: int
    # model training parameters
    surrogate_learning_rate: float
    surrogate_batch_size: int
    num_model_training_steps: int
    surrogate_replay_buffer_size: int
    surrogate_model_update_period: int
    max_epochs_since_improvement: int
    
    ts_inf: bool
    prob: bool
    learn_std: bool
    fixed_std: float
    use_grad_clipping: bool
    grad_clip_value: float
    num_batches_per_loss: int
    learn_rwd: bool

@hydra.main(config_path="configs/mapelites", config_name="daqd")
def train(config: ExperimentConfig) -> None:
    # setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")
    output_dir = "./"#get_output_dir()
    _last_metrics_dir = os.path.join(output_dir, "checkpoints", "last_metrics")
    _last_grid_dir = os.path.join(output_dir, "checkpoints", "last_grid")
    _grid_img_dir = os.path.join(output_dir, "images", "me_grids")
    _metrics_img_dir = os.path.join(output_dir, "images", "me_metrics")
    _timings_dir = os.path.join(output_dir, "timings")
    _init_state_dir = os.path.join(output_dir, "init_state")
    os.makedirs(_last_metrics_dir, exist_ok=True)
    os.makedirs(_last_grid_dir, exist_ok=True)
    os.makedirs(_grid_img_dir, exist_ok=True)
    os.makedirs(_metrics_img_dir, exist_ok=True)
    os.makedirs(_timings_dir, exist_ok=True)
    os.makedirs(_init_state_dir, exist_ok=True)

    # Init environment
    env_name = config.env_name
    env = environments.create(env_name)
    print("Observation size: ",env.observation_size)
    print("Action size: ",env.action_size)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        activation=flax.linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Save initial state
    with open(os.path.join(_init_state_dir, "init_states.pkl"), "wb") as file_to_save:
        init_state = jax.tree_util.tree_map(
                    lambda x: x[0],
                    init_states
                    )
        pickle.dump(init_state, file_to_save)  

    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        reset_based_scoring_function,
        play_reset_fn=lambda random_key: init_state,
        episode_length=config.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Define emitter
    variation_fn = functools.partial(isoline_variation, 
                                     iso_sigma=config.iso_sigma, 
                                     line_sigma=config.line_sigma)
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config.batch_size,
    )


    # Define surrogate model config
    surrogate_model_config = DynamicsModelConfig(
        imagination_horizon=config.episode_length,
        surrogate_hidden_layer_sizes=config.surrogate_hidden_layer_sizes,
        surrogate_ensemble_size=config.surrogate_ensemble_size,
        surrogate_learning_rate=config.surrogate_learning_rate,
        surrogate_batch_size=config.surrogate_batch_size,
        num_model_training_steps=config.num_model_training_steps,
        surrogate_replay_buffer_size=config.surrogate_replay_buffer_size,
        surrogate_model_update_period=config.surrogate_model_update_period,
        num_imagined_iterations =config.num_imagined_iterations,
        add_buffer_size=config.add_buffer_size,
        max_epochs_since_improvement=config.max_epochs_since_improvement,
        prob=config.prob,
        learn_std=config.learn_std,
        fixed_std=config.fixed_std,
        use_grad_clipping=config.use_grad_clipping,
        grad_clip_value=config.grad_clip_value,
        num_batches_per_loss=config.num_batches_per_loss,
        ts_inf=config.ts_inf,
    )

    # Define reward and bd extractor for imagination
    if env_name not in fitness_extractor_imagination:
        raise NotImplementedError(f"Dynamics model/World model does not support {env_name} yet.")
    imagined_reward_extractor_fn = fitness_extractor_imagination[env_name]
    imagined_bd_extractor_fn = bd_extractor_imagination[env_name]

    # # Define the surrogate model 
    surrogate_model = DynamicsModelEnsemble(
        config=surrogate_model_config,
        policy_network=policy_network,
        state_size=env.observation_size,
        action_size=env.action_size,
        play_reset_fn=lambda random_key: init_state.obs,
        reward_extractor_fn=imagined_reward_extractor_fn,
        bd_extractor_fn=imagined_bd_extractor_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict:
        # Get metrics
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        # Add offset for positive qd_score
        qd_score += reward_offset * config.episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": jnp.array([qd_score]), "max_fitness": jnp.array([max_fitness]), "coverage": jnp.array([coverage])}

    #print("Init states: ",init_states.obs.shape)
    # Instantiate MAP-Elites
    map_elites = ModelBasedMAPElites(
        surrogate_model=surrogate_model,
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
        surrogate_config=surrogate_model_config,
    )

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids = compute_euclidean_centroids(
        grid_shape=config.grid_shape,
        minval=minval,
        maxval=maxval,
    )
    duration = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {duration:.2f}s")

    # Init algorithm
    num_iterations = (config.num_evaluations // config.batch_size) + 1 # overwrite num iterations with num_evaluations
    logger.warning("--- Algorithm initialisation ---")
    total_training_time = 0.0
    start_time = time.time()
    repertoire, imagined_repertoire, emitter_state, surrogate_state, random_key = map_elites.init(init_variables, centroids, random_key)
    init_time = time.time() - start_time
    total_training_time += init_time
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")
    current_step_estimation = 0
    full_metrics = {"coverage": jnp.array([0.0]), 
                    "max_fitness": jnp.array([0.0]), 
                    "qd_score": jnp.array([0.0])}
    timings = {"init_time": init_time,
               "centroids_time": duration,
               "runtime_logs": jnp.zeros([(num_iterations)+1, 1]),
               "avg_iteration_time": 0.0,
               "avg_evalps": 0.0}

    # Main QD Loop
    for iteration in range(num_iterations):
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_iterations} ---"
        )

        start_time = time.time()
        (repertoire, imagined_repertoire, emitter_state, surrogate_state, metrics, random_key,) = map_elites.update(
            repertoire,
            imagined_repertoire,
            emitter_state,
            surrogate_state,
            random_key,
        )

        time_duration = time.time() - start_time # time for log_period iterations
        total_training_time += time_duration

        logger.warning(f"--- Current QD Score: {metrics['qd_score'][-1]:.2f}")
        logger.warning(f"--- Current Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning(f"--- Current Max Fitness: {metrics['max_fitness'][-1]}")
        
        if iteration % config.surrogate_model_update_period == 0:
            print("Training model")
            train_model_time = time.time()
            surrogate_state = map_elites._surrogate_model.train_model(
                surrogate_state=surrogate_state,
            )
            print("Surrogate state train loss: ", surrogate_state.loss)
            print("Train model time: ", time.time() - train_model_time)
            
        # Save metrics
        full_metrics = {key: jnp.concatenate((full_metrics[key], metrics[key])) for key in full_metrics}
        # Save timings
        timings["avg_iteration_time"] = (timings["avg_iteration_time"]*(iteration*config.log_period) + time_duration) / ((iteration+1)*config.log_period)
        timings["avg_evalps"] = (timings["avg_evalps"]*(iteration*config.log_period) + ((config.batch_size*config.log_period)/time_duration)) / ((iteration+1)*config.log_period)
        timings["runtime_logs"] = timings["runtime_logs"].at[iteration, 0].set(total_training_time)   
        if iteration%config.log_period == 0: 
            with open(os.path.join(_last_metrics_dir, "metrics.pkl"), "wb") as file_to_save:
                pickle.dump(full_metrics, file_to_save)
            # Save timings
            with open(os.path.join(_timings_dir, "runtime.pkl"), "wb") as file_to_save:
                pickle.dump(timings, file_to_save)    
        
        # Save repertoire map
        if (config.plot_grid == True and iteration%config.plot_grid_log_period == 0) and env.behavior_descriptor_length == 2:
            fig, ax = plot_2d_map_elites_repertoire(
                centroids,
                repertoire.fitnesses,
                minval,
                maxval,
                repertoire.descriptors,
            )
            fig.savefig(os.path.join(_grid_img_dir, f"grid_{iteration}"))
            plt.close(fig)

            # create the plots for metrics
            num_evals = jnp.arange(iteration+2) * config.batch_size # +2 because one for the zero, which we init at zero and one for the first iteration
            fig, axes = plot_map_elites_results(env_steps=num_evals, metrics=full_metrics, repertoire=repertoire, min_bd=minval, max_bd=maxval)
            fig.savefig(os.path.join(_metrics_img_dir, f"metrics_{iteration}"))
            plt.close(fig)

        # Store the latest controllers of the repertoire
        if config.store_repertoire == True and iteration%config.store_repertoire_log_period == 0:
            repertoire.save(path=_last_grid_dir+"/")


    duration = time.time() - init_time

    logger.warning("--- Final metrics ---")
    logger.warning(f"Duration: {duration:.2f}s")
    logger.warning(f"Training duration: {total_training_time:.2f}s")
    logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")

    # Save final metrics
    with open(os.path.join(_last_metrics_dir, "metrics.pkl"), "wb") as file_to_save:
                pickle.dump(full_metrics, file_to_save)
    # Save final repertoire
    repertoire.save(path=_last_grid_dir+"/")


if __name__ == "__main__":
    train()
