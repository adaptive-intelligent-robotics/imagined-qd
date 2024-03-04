from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import jax
import flax.linen as nn
import optax

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from src.models.utils import DataBuffer, Datapoint
from models.base_modules import (
    DirectModule, 
    make_direct_model_loss_fn,
    ProbDirectModule, 
    make_prob_direct_model_loss_fn,
)

from models.base_model import SurrogateModel, SurrogateModelState, SurrogateModelConfig

@dataclass
class DirectModelConfig(SurrogateModelConfig):

    # model parameters
    surrogate_hidden_layer_sizes: Tuple[int, ...] = (128, 128)
    surrogate_ensemble_size: int = 1

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
    surrogate_data_buffer_size: int = 1000000
    surrogate_model_update_period: int = 10 # how often to update the model
    max_epochs_since_improvement: int = 5
    

class DirectModelState(SurrogateModelState):

    params: Params
    optimizer_state: optax.OptState
    data_buffer: DataBuffer
    random_key: RNGKey
    loss: float


class DirectModel(SurrogateModel):

    def __init__(
        self,
        config: DirectModelConfig,
        genotype_size: int,
        desc_size: int,
    ):
        self._config = config

        self._genotype_size = genotype_size
        self._desc_size = desc_size

        # Probabilistic or Deterministic Direct Model
        if self._config.prob:
            
            # init prob dynamics model network
            if self._config.learn_std:
                self._output_size = (self._desc_size + 1) * 2
            else:
                self._output_size = self._desc_size + 1

            self.direct_model = ProbDirectModule(
                    input_size=self._genotype_size,
                    output_size=self._output_size,
                    hidden_layer_sizes=self._config.surrogate_hidden_layer_sizes,
                    input_mu=jnp.zeros(shape=(self._genotype_size,)),
                    input_std=jnp.ones(shape=(self._genotype_size,)),
                    output_mu=jnp.zeros(shape=(self._output_size,)),
                    output_std=jnp.ones(shape=(self._output_size,)),
                    learn_std=self._config.learn_std,
                    fixed_std=self._config.fixed_std,
            )

            self._direct_model_loss_fn = make_prob_direct_model_loss_fn(direct_model_fn=self.direct_model.apply,
                                                                        learn_std=self._config.learn_std,
                                                                        fixed_std=self._config.fixed_std)

        else:
            # init dynamics model network
            self.direct_model = DirectModule(
                input_size=self._genotype_size,
                output_size=self._output_size,
                hidden_layer_sizes=self._config.surrogate_hidden_layer_sizes,
                input_mu=jnp.zeros(shape=(self._genotype_size,)),
                input_std=jnp.ones(shape=(self._genotype_size,)),
                output_mu=jnp.zeros(shape=(self._output_size,)),
                output_std=jnp.ones(shape=(self._output_size,)),
            )

            # init loss function
            self._direct_model_loss_fn = make_direct_model_loss_fn(direct_model_fn=self.direct_model.apply)

        # init optimizer
        if self._config.use_grad_clipping:
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.grad_clip_value),
                optax.adam(learning_rate=self._config.surrogate_learning_rate),
            )
        else:
            self._optimizer = optax.adam(learning_rate=self._config.surrogate_learning_rate)

        

    def init(self, random_key: RNGKey) -> Tuple[DirectModelState, RNGKey]:
        """
        Initializes the training state (model params and optimizer state) of the model
        """
        # init direct model params
        random_key, subkey = jax.random.split(random_key)
        fake_genotype = jnp.zeros(shape=(self._genotype_size,))
        init_params = self.direct_model.init(subkey, fake_genotype)

        # init optimizer
        optimizer_state = self._optimizer.init(init_params)
        
        # initialize data buffer
        dummy_data = Datapoint.init_dummy(
            genotype_dim=self._genotype_size,
            desc_dim=self._desc_size,
        )

        data_buffer = DataBuffer.init(
            buffer_size=self._config.surrogate_data_buffer_size, transition=dummy_data
        )

        # initialize training state
        random_key, subkey = jax.random.split(random_key)
        training_state = DirectModelState(params=init_params, 
                                          optimizer_state=optimizer_state,
                                          data_buffer=data_buffer,
                                          random_key=subkey,
                                          loss=jnp.inf)

        return training_state, random_key


    @partial(jax.jit, static_argnames=("self",))
    def scoring_function(
        self, 
        params: Genotype, 
        random_key: RNGKey,
        model_params: Params,
        ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        Returns the expected fitness and descriptors of the genotypes provided as computed by the direct model.
        """

        # Perform rollouts with each policy
        random_key, prob_key = jax.random.split(random_key)

        get_pred_dist_fn = partial(self.direct_model.get_pred_dist, model_params)

        prob_keys = jax.random.split(prob_key, params.shape[0])
        fit_bd_mean, fit_bd_std, samples = jax.vmap(get_pred_dist_fn)(params, prob_keys)

        fitnesses = fit_bd_mean[:, 0]
        descriptors = fit_bd_mean[:, 1:]
        
        datapoints = Datapoint(
            genotype=params,
            fitness=fitnesses,
            desc=descriptors,
        )

        # print("Fitnesses shape: ", fitnesses.shape)
        # print("Descriptors shape: ", descriptors.shape)

        return (
            fitnesses,
            descriptors,
            {"mean_fit_pred": fit_bd_mean[:, 0],
             "std_fit_pred": fit_bd_std[:, 0],
             "mean_desc_pred": fit_bd_mean[:, 1:],
             "std_desc_pred": fit_bd_std[:, 1:],
             "samples_fit": samples[:, 0],
             "samples_desc": samples[:, 1:],
             "datapoints": datapoints, 
            },
            random_key,
        )


    def update(
        self,
        surrogate_state: DirectModelState, 
        extra_scores: ExtraScores,
    ) -> DirectModelState:
        """
        Update the buffer of the surrogate model
        """
        datapoints = extra_scores["datapoints"]

        # add datapoints in the data buffer
        data_buffer = surrogate_state.data_buffer.insert(datapoints)
        surrogate_state = surrogate_state.replace(data_buffer=data_buffer)

        return surrogate_state

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

        data_buffer = surrogate_state.data_buffer
        model_params = surrogate_state.params
        optimizer_state = surrogate_state.optimizer_state

        _loss_fn = partial(self._direct_model_loss_fn, output_mu=self.direct_model.output_mu, output_std=self.direct_model.output_std)
        n_train = train_data.shape[0]
        num_batches = self._config.num_batches_per_loss #n_train // self._config.surrogate_batch_size
        for it in range(num_batches):
            # Sample a batch of transitions in the train set - batchsize
            samples, random_key = data_buffer.sample_data(
                random_key, train_data, sample_size=self._config.surrogate_batch_size
            )
            
            # Compute the loss and update the model
            loss, gradient = jax.value_and_grad(_loss_fn)(
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

            model_params = optax.apply_updates(model_params, model_updates)
            # print("Model params shape: ", jax.tree_util.tree_map(lambda x: x.shape, model_params))
            # print(f"Training Loss {it}/{num_batches}: ", loss)

        training_steps = training_steps + num_batches

        # Compute the test loss
        test_samples, random_key = data_buffer.sample_data(
            random_key, test_data, sample_size=self._config.surrogate_batch_size
        )
        # print("Test samples obs shape: ", jax.tree_util.tree_map(lambda x: x.shape, test_samples))

        test_loss, _ = jax.lax.stop_gradient(
            jax.value_and_grad(_loss_fn)(
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

    def train_model(
        self,
        surrogate_state: DirectModelState
    ) -> DirectModelState:
        """
        Trains the surrogate model.
        """

        random_key = surrogate_state.random_key
        data_buffer = surrogate_state.data_buffer
        
        all_data= data_buffer.get_all_data()
        assert not jnp.any(jnp.isnan(all_data.genotype)), "NaNs in genotypes"
        self.direct_model.fit_input_stats(all_data.genotype)
        self.direct_model.fit_output_stats(jnp.expand_dims(all_data.fitness, axis=-1), all_data.desc)

        # split data into train and test sets
        train_data, test_data, random_key = data_buffer.train_test_split(random_key)

        # train model for n steps or until test loss stagnates
        with jax.disable_jit():
            (_, _, new_surrogate_state, random_key, _, _, _) = jax.lax.while_loop(
                self._train_model_cond_fun, 
                self._train_model_body_fun,
                (train_data, test_data, surrogate_state, random_key, 10e6, 0, 0)
            )

        # (new_surrogate_state, random_key) = self._train_model_simple(surrogate_state, random_key)

        # Create new training state
        new_training_state = DirectModelState(
            params=new_surrogate_state.params,
            optimizer_state=new_surrogate_state.optimizer_state,
            data_buffer=data_buffer,
            random_key=random_key,
            loss=new_surrogate_state.loss
        )

        return new_training_state
