from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
import optax
from flax.struct import PyTreeNode

from qdax.types import EnvState, Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

'''
Surrogate model class/object in QD is anything that will be used to predict the fitness and feature of a set of params
- It can be a neural network, a gaussian process, a linear model, etc.
- mainly neural networks of different forms implemented in this library
'''

class SurrogateModelState(PyTreeNode):
    """
    State of the surrogate model
    """
    pass

@dataclass
class SurrogateModelConfig:
    """
    Configuration for the surrogate model
    """
    pass

class SurrogateModel:
    '''
    defines the base class for all surrogate models for QD
    '''

    def __init__(self,
                 config: SurrogateModelConfig
                 ):
        
        self._config = config

    def init(self, random_key: RNGKey) -> Tuple[SurrogateModelState, RNGKey]:
        """
        Initializes the model and returns the initial state of the model
        """
        raise NotImplementedError
    
    def scoring_function(self, 
        params: Genotype, 
        random_key: RNGKey,
        model_params: Params,
        ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        main function of the surrogate model
        Returns the expected fitness and features/descriptors of the genotypes/params provided as computed by the surrogate model.
        """

        raise NotImplementedError
    
    def update(self, params: PyTreeNode, grads: PyTreeNode) -> PyTreeNode:
        """
        Updates the surrogate model state
        """
        raise NotImplementedError
    
    def train_model(self, 
                    params: PyTreeNode, 
                    grads: PyTreeNode, 
                    optimizer: optax.GradientTransformation,
                    learning_rate: float) -> PyTreeNode:
        """
        Trains the model
        """
        raise NotImplementedError