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
