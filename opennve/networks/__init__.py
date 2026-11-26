"""Neural networks: architecture registry, Network (Keras model), NeuralEngine, TensorFlow Lite export.

TensorFlow is imported only when a network is built or loaded. Built-in architecture: "UNet".
New architectures: register_network("Name")(builder); custom Keras layers for loading saved
models: register_custom_objects({...}); weight transfers: register_weight_transfer("name")(function);
weights of any source into any network of the same layers: transfer_weights(source, destination);
Lambda layers of saved models that load with any Python version: register_lambda(pattern, layer).
"""

from .engine import NeuralEngine
from .network import (
    QUANTIZATIONS,
    Network,
    WeightComparison,
    WeightDifference,
    compare_weights,
    load_keras_model,
    register_weight_transfer,
    representative_dataset,
    transfer_weights_by_name,
)
from .lambdas import register_lambda, unregister_lambda
from .registry import (
    NetworkParameters,
    build_network,
    custom_objects,
    list_networks,
    register_custom_objects,
    register_network,
    unregister_network,
)
from .transfer import TransferReport, read_weights, transfer_weights

__all__ = [
    "NeuralEngine",
    "Network",
    "NetworkParameters",
    "QUANTIZATIONS",
    "WeightComparison",
    "WeightDifference",
    "build_network",
    "compare_weights",
    "custom_objects",
    "list_networks",
    "load_keras_model",
    "register_custom_objects",
    "register_network",
    "register_weight_transfer",
    "representative_dataset",
    "transfer_weights_by_name",
    "unregister_network",
    "TransferReport",
    "read_weights",
    "register_lambda",
    "transfer_weights",
    "unregister_lambda",
]
