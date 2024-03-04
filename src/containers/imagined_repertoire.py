from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.types import (
    Centroid,
    Descriptor,
    Fitness,
    Genotype,
    Mask,
    ParetoFront,
    RNGKey,
)


class ImaginedRepertoire(MapElitesRepertoire):
    """Class for the imagined repertoire in Model Based Map Elites

    This class inherits from MAPElitesRepertoire. The stored data
    is the same: genotypes, fitnesses, descriptors, centroids adds additional stored data.
    Additional stored data is the addition buffer keeping track of solutions that have just been added

    Inherited functions: sample, save and load.
    """

    add_buffer: Genotype 
    add_buffer_size: int
    add_buffer_position: int = 0
    

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> MapElitesRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)

        Returns:
            The updated MAP-Elites repertoire.
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        # get newly added genotypes and append to current add_buffer
        num_added_solutions = jnp.count_nonzero(addition_condition)
        # gets the indices of the batch of added solutions and to maintain fixed size of batch fills the rest with a large index
        add_indices = jnp.nonzero(addition_condition, size=addition_condition.shape[0], fill_value=num_centroids)[0]
        #print("add indices shape: ", add_indices.shape)
        add_indices_sorted = jnp.sort(add_indices, axis=0)

        added_genotypes = jax.tree_map(lambda x: x.at[add_indices_sorted].get(), batch_of_genotypes)
        # print("addition_condition: ", addition_condition.shape)
        # print("batch_of_genotypes: ", jax.tree_map(lambda x: x.shape, batch_of_genotypes))
        #print("added_genotypes: ", jax.tree_map(lambda x: x.shape, added_genotypes))
        # print("add buffer: ", jax.tree_map(lambda x: x.shape, self.add_buffer))
        # print("num_added_solutions: ", num_added_solutions)
        # print("new add buffer position: ", self.add_buffer_position+num_added_solutions)
        new_add_buffer = jax.tree_map(
            lambda add_buffer, new_genotypes: jax.lax.dynamic_update_slice_in_dim(add_buffer, new_genotypes, start_index=self.add_buffer_position, axis=0),
            self.add_buffer,
            added_genotypes,
        )

        return ImaginedRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            add_buffer=new_add_buffer,
            add_buffer_size=self.add_buffer_size,
            add_buffer_position=self.add_buffer_position+num_added_solutions,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        add_buffer_size: int,
    ) -> ImaginedRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize repertoire with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        
        default_add_buffer = jax.tree_map(
            lambda x: jnp.zeros(shape=(add_buffer_size*2,) + x.shape[1:]),
            genotypes,
        )

        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            add_buffer=default_add_buffer,
            add_buffer_size=add_buffer_size,
        )

        # Add initial values to the repertoire
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return new_repertoire  # type: ignore

    def select(self) -> Genotype:
        """
        THIS SELECTS ALL
        In this particular case, we select the genotypes that have just been recently added to the imagined repertoire.
        More generally, we can write your own selection function/procedure to select from imagined repertoire.
        """ 

        selected_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[:self.add_buffer_size].get(),
            self.add_buffer
        )

        return selected_genotypes

    def select_sample(self, random_key, num_samples) -> Genotype:
        """
        In this function, we sample a batch of genotypes from the add buffer

        !!!WARNING!!! Non jitable as currently implemented
        """ 

        if self.add_buffer_position == 0:
            return jax.tree_util.tree_map(lambda x: x.at[:num_samples].get(), self.add_buffer), random_key

        # Get current content of buffer
        all_add_buffer = jax.tree_util.tree_map(
            lambda x: x.at[:self.add_buffer_position].get(),
            self.add_buffer
        )

        # Shuffle content
        random_key, subkey = jax.random.split(random_key)
        shuffled_add_buffer = jax.tree_util.tree_map(
            lambda x: jax.random.permutation(subkey, x, axis=0),
            all_add_buffer,
        )

        # Get num_samples individuals 
        # at[].get() handle out of bound by returning last indiv multiple times
        # So work even if self.add_buffer_position < num_samples
        indexes = jnp.arange(0, num_samples, step=1)
        samples = jax.tree_util.tree_map(
            lambda x: x.at[indexes].get(),
            shuffled_add_buffer,
        )
        
        return samples, random_key


    def sync_repertoire(self, repertoire: MapElitesRepertoire) -> ImaginedRepertoire:
        """
        Synchronize the imagined repertoire with the real repertoire.
        """
        repertoire = self.replace(
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.descriptors,
        )
        return repertoire

    def clear_add_buffer(self) -> ImaginedRepertoire:

        default_add_buffer = jax.tree_map(
            lambda x: jnp.zeros(shape=(self.add_buffer_size*2,) + x.shape[1:]),
            self.genotypes,
        )
        repertoire = self.replace(add_buffer=default_add_buffer,
                                  add_buffer_position=0)
        
        return repertoire
