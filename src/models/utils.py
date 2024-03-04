from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import RNGKey

from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer


class ImprovedReplayBuffer(ReplayBuffer):
    '''
    Improved Replay Buffer with more utilities for training models 
    - like splitting buffer for train and test split etc.
    '''

    def get_all_transitions(self) -> jnp.ndarray:
        """
        Returns:
            all transitions in the buffer.
        """
        a = 0
        b = self.current_size
        samples = self.data.at[a:b].get()
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions

    def train_test_split(self, random_key: RNGKey, test_size: float=0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            a tuple of train and test transitions.
        """
        random_key, subkey = jax.random.split(random_key)
        all_data = self.data.at[:self.current_size].get()
        shuffled_data = jax.random.permutation(subkey, all_data, axis=0)
        train_data = shuffled_data[:int(self.current_size * (1 - test_size))]
        test_data = shuffled_data[int(self.current_size * (1 - test_size)) :]

        return train_data, test_data, random_key

    def sample_data(self, random_key: RNGKey, data: jnp.ndarray, sample_size: int) -> jnp.ndarray:
        """
        Samples from given flattened data array - as opposed to just sampling from the replay buffer
        Returns:
            a batch of transitions.
        """
        random_key, subkey = jax.random.split(random_key)
        
        # shuffled_data = jax.random.shuffle(subkey, data)
        # samples = shuffled_data.at[:sample_size].get()
        
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=data.shape[0],
        )
        samples = jnp.take(data, idx, axis=0, mode="clip")

        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    def clean_up_buffer(self):
        """Clean up nan transition in the replay buffer before current_size."""
        data = self.data
        position = jnp.arange(0, self.buffer_size, step=1)
        condition = jnp.logical_and(position < self.current_size, jnp.any(jnp.isnan(data[position]), axis=1))
        if jnp.any(condition):
            print("Found NaN in replay buffer:", data[condition])
        data = jnp.where(
            jnp.repeat(jnp.expand_dims(condition, axis=1), data.shape[1], axis=1),
            data[0],
            data,
        )
        return self.replace(data=data) # type: ignore


class Datapoint(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    genotype: jnp.ndarray
    fitness: jnp.ndarray
    desc: jnp.ndarray

    @property
    def genotype_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.genotype.shape[-1]  # type: ignore

    @property
    def desc_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.desc.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the datapoint once flattened.

        """
        flatten_dim = self.genotype_dim + self.desc_dim + 1
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.genotype,
                jnp.expand_dims(self.fitness, axis=-1),
                self.desc,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_datapoint: jnp.ndarray,
        datapoint: Datapoint,
    ) -> Datapoint:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        genotype_dim = datapoint.genotype_dim
        desc_dim = datapoint.desc_dim

        genotype = flattened_datapoint[:, :genotype_dim]
        fitness = jnp.ravel(flattened_datapoint[:, genotype_dim : (genotype_dim + 1)])
        desc = flattened_datapoint[:, (genotype_dim + 1):]
        

        return cls(
            genotype=genotype,
            fitness=fitness,
            desc=desc,
        )

    @classmethod
    def init_dummy(cls, genotype_dim: int, desc_dim: int) -> Datapoint:
        """
        Initialize a dummy datapoint that then can be passed to constructors to get
        all shapes right.
        """
        dummy_transition = Datapoint(
            genotype=jnp.zeros(shape=(1, genotype_dim)),
            fitness=jnp.zeros(shape=(1,)),
            desc=jnp.zeros(shape=(1, desc_dim)),
        )
        return dummy_transition
    


class DataBuffer(flax.struct.PyTreeNode):
    """
    A replay buffer where transitions are flattened before being stored.
    Transitions are unflatenned on the fly when sampled in the buffer.
    data shape: (buffer_size, transition_concat_shape)
    """

    data: jnp.ndarray
    buffer_size: int = flax.struct.field(pytree_node=False)
    transition: Datapoint

    current_position: jnp.ndarray = flax.struct.field()
    current_size: jnp.ndarray = flax.struct.field()

    @classmethod
    def init(
        cls,
        buffer_size: int,
        transition: Datapoint,
    ) -> DataBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.

        Args:
            buffer_size: the size of the replay buffer, e.g. 1e6
            transition: a transition object (might be a dummy one) to get
                the dimensions right
        """
        flatten_dim = transition.flatten_dim
        data = jnp.ones((buffer_size, flatten_dim)) * jnp.nan
        current_size = jnp.array(0, dtype=int)
        current_position = jnp.array(0, dtype=int)
        return cls(
            data=data,
            current_size=current_size,
            current_position=current_position,
            buffer_size=buffer_size,
            transition=transition,
        )

    @partial(jax.jit, static_argnames=("sample_size",))
    def sample(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Datapoint, RNGKey]:
        """
        Sample a batch of transitions in the replay buffer.
        """
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        samples = jnp.take(self.data, idx, axis=0, mode="clip")
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    @jax.jit
    def insert(self, transitions: Datapoint) -> DataBuffer:
        """
        Insert a batch of transitions in the replay buffer. The transitions are
        flattened before insertion.

        Args:
            transitions: A transition object in which each field is assumed to have
                a shape (batch_size, field_dim).
        """
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1])
        )
        num_transitions = flattened_transitions.shape[0]
        max_replay_size = self.buffer_size

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > max_replay_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {num_transitions}, "
                f"max replay size {max_replay_size}"
            )

        # get current position
        position = self.current_position

        # check if there is an overlap
        roll = jnp.minimum(0, max_replay_size - position - num_transitions)

        # roll the data to avoid overlap
        data = jnp.roll(self.data, roll, axis=0)

        # update the position accordingly
        new_position = position + roll

        # replace old data by the new one
        new_data = jax.lax.dynamic_update_slice_in_dim(
            data,
            flattened_transitions,
            start_index=new_position,
            axis=0,
        )

        # update the position and the size
        new_position = (new_position + num_transitions) % max_replay_size
        new_size = jnp.minimum(self.current_size + num_transitions, max_replay_size)

        # update the replay buffer
        replay_buffer = self.replace(
            current_position=new_position,
            current_size=new_size,
            data=new_data,
        )

        return replay_buffer  # type: ignore
    
    def get_all_data(self) -> jnp.ndarray:
        """
        Returns:
            all data in the buffer.
        """
        a = 0
        b = self.current_size
        samples = self.data.at[a:b].get()
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions

    def train_test_split(self, random_key: RNGKey, test_size: float=0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            a tuple of train and test transitions.
        """
        random_key, subkey = jax.random.split(random_key)
        all_data = self.data.at[:self.current_size].get()
        shuffled_data = jax.random.permutation(subkey, all_data, axis=0)
        train_data = shuffled_data[:int(self.current_size * (1 - test_size))]
        test_data = shuffled_data[int(self.current_size * (1 - test_size)) :]

        return train_data, test_data, random_key
    
    def sample_data(self, random_key: RNGKey, data: jnp.ndarray, sample_size: int) -> jnp.ndarray:
        """
        Samples from given flattened data array
        Returns:
            a batch of transitions.
        """
        random_key, subkey = jax.random.split(random_key)
        
        # shuffled_data = jax.random.shuffle(subkey, data)
        # samples = shuffled_data.at[:sample_size].get()
        
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=data.shape[0],
        )
        samples = jnp.take(data, idx, axis=0, mode="clip")

        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key


    def clean_up_buffer(self):
        """Clean up nan transition in the replay buffer before current_size."""
        data = self.data
        position = jnp.arange(0, self.buffer_size, step=1)
        condition = jnp.logical_and(position < self.current_size, jnp.any(jnp.isnan(data[position]), axis=1))
        if jnp.any(condition):
            print("Found NaN in replay buffer:", data[condition])
        data = jnp.where(
            jnp.repeat(jnp.expand_dims(condition, axis=1), data.shape[1], axis=1),
            data[0],
            data,
        )
        return self.replace(data=data) # type: ignore

