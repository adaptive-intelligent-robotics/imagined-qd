"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple
import logging

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from src.containers.imagined_repertoire import ImaginedRepertoire
from src.models.dynamics_model import DynamicsModel, SurrogateModelState, SurrogateModelConfig


class ModelBasedMAPElites:
    """Core elements of Model-based QD (MAP-Elites) algorithms.

    Args:
        surrogate_model: a function that takes a batch of genotypes and returns
            expected fitnesses and descriptors
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors on the task
        emitter: an emitter/generator is used to generate new offsprings given a QD archive (MAPELites
            repertoire). It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any desired metrics to track evolution and progress
    """

    def __init__(
        self,
        surrogate_model: DynamicsModel,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        surrogate_config: SurrogateModelConfig,
    ) -> None:

        self._surrogate_model = surrogate_model
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._config = surrogate_config

    #@partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, ImaginedRepertoire, Optional[EmitterState], Optional[SurrogateModelState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes - extra scores contains transition trajectories
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
        )

        # init the imagined archive 
        imagined_repertoire = ImaginedRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            add_buffer_size=self._config.add_buffer_size,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # get initial state of the surrogate model
        surrogate_model_state, random_key = self._surrogate_model.init(
            random_key=random_key
        )

        # update surrogate model state using evaluations/tranjectories performed during the initialization
        surrogate_model_state = self._surrogate_model.update(
            surrogate_state=surrogate_model_state,
            extra_scores=extra_scores,
        )   
        
        # train the surrogate model
        print("Initial training the surrogate model")
        new_surrogate_model_state = self._surrogate_model.train_model(
            surrogate_state=surrogate_model_state,
        )

        return repertoire, imagined_repertoire, emitter_state, new_surrogate_model_state, random_key

    #@partial(jax.jit, static_argnames=("self",))
    def update_imagination(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        surrogate_model_state: Optional[SurrogateModelState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm in imagination.
        1. A batch of genotypes is sampled from the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the imagined repertoire.

        Args:
            repertoire: the MAP-Elites repertoire (could be real or imagined)
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # scores the offsprings with the surrogate model
        imagined_scoring_fn = partial(
            self._surrogate_model.scoring_function, 
            model_params=surrogate_model_state.params,
        )
        fitnesses, descriptors, extra_scores, random_key = imagined_scoring_fn(
            genotypes,
            random_key,
        )

        # add genotypes in the (imagined) repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics (QD metrics in imagination)
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key

    def _update_body_fun(self, state):
        """
        Body function of the update loop.
        """
        (
            repertoire,
            imagined_repertoire,
            emitter_state,
            surrogate_model_state,
            random_key,
            it,
        ) = state

        # update the imagined repertoire
        imagined_repertoire, emitter_state, metrics, random_key = self.update_imagination(
            repertoire=imagined_repertoire,
            emitter_state=emitter_state,
            surrogate_model_state=surrogate_model_state,
            random_key=random_key,
        )

        it = it + 1
        return (
            repertoire,
            imagined_repertoire,
            emitter_state,
            surrogate_model_state,
            random_key,
            it,
        )


    def _update_cond_fun(self, state):
        """
        Condition function of the update loop.
        Termination condition of the imagination loop
        """
        (
            repertoire,
            imagined_repertoire,
            emitter_state,
            surrogate_model_state,
            random_key,
            it,
        ) = state
        
        #print("Imagined iteration {}".format(it))
        #print("Add buffer position: ", imagined_repertoire.add_buffer_position)
        #print("Add buffer size: ", self._config.add_buffer_size)
        # print("cond 1: ", imagined_repertoire.add_buffer_position < self._config.add_buffer_size)
        # print("cond 2: ", it < self._config.num_imagined_iterations)
        return jnp.logical_and(
            imagined_repertoire.add_buffer_position < self._config.add_buffer_size,
            it < self._config.num_imagined_iterations
        )



    #@partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        imagined_repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        surrogate_model_state: SurrogateModelState,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, ImaginedRepertoire, Optional[EmitterState], SurrogateModelState, Metrics, RNGKey]:
        """
        Performs:
        1. Selection from the imagined repertoire
        2. Execution of the solution from the imagined repertoire
        3. Addition of the solutions in the the real repertoire

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # perform n iterations of the MAP-Elites algorithm in imagination (until threhold or until n)
        # while imagined_repertoire.add_buffer_position < self._config.threshold_add_buffer:
        # for _ in range(self._config.num_imagined_iterations):
        #     (imagined_repertoire, emitter_state, metrics, random_key,) = self.update_imagination(
        #         imagined_repertoire, 
        #         emitter_state, 
        #         surrogate_model_state,
        #         random_key
        #     )
        print("Imagination loop")
        (repertoire, imagined_repertoire, emitter_state, surrogate_model_state, random_key, im_it) = jax.lax.while_loop(
            self._update_cond_fun, 
            self._update_body_fun, 
            (repertoire, imagined_repertoire, emitter_state, surrogate_model_state, random_key, 0)
        )
        print("Number of imagined iterations:", im_it)
        print("Num of solutions added to imagined buffer: ", imagined_repertoire.add_buffer_position)

        # select solutions from the imagined repertoire
        genotypes = imagined_repertoire.select()

        # scores the selected solutions (evaluation)
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire (addition)
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        # update the metrics
        metrics = self._metrics_function(repertoire)

        # update surrogate state with transitions
        surrogate_model_state = self._surrogate_model.update(
            surrogate_state=surrogate_model_state,
            extra_scores=extra_scores,
        ) 

        # train the surrogate model - placed outside now
        # if self._config.surrogate_model_update_period:
        # print("Training the surrogate model")
        # surrogate_model_state = self._surrogate_model.train_model(
        #     surrogate_state=surrogate_model_state,
        # )

        # reset the imagined repertoire to be the latest repertoire and clear the buffer
        imagined_repertoire = imagined_repertoire.sync_repertoire(repertoire)
        imagined_repertoire = imagined_repertoire.clear_add_buffer()

        return repertoire, imagined_repertoire, emitter_state, surrogate_model_state, metrics, random_key  

    @partial(jax.jit, static_argnames=("self",))
    def update_normal(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key


    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, emitter_state, random_key = carry
        (repertoire, emitter_state, metrics, random_key,) = self.update(
            repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, emitter_state, random_key), metrics
