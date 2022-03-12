Algorithm trainer specification
===============================

An algorithm trainer class exposes at least the following API

- ``__init__(self, config: Config, algorithm: Algorithm)``
- ``train(self, env: Env, **options) -> TrainInfo``

The constructor ``Config`` parameter specifies the configuration options for the trainer e.g. number of episodes or number of
workers. The ``Algorithm`` parameter specifies the algorithm to train. Training occurs when the ``train`` method is called
with the environment instance that should be used for training. An instance of the ``TrainInfo`` is returned to the calling
site providing basic information about the training

Currently, there are two classes that implement the above API:

- The ``RLSerialAlgorithmTrainer`` class
- The ``PyTorchParallelTrainer`` class

The `RLSerialAlgorithmTrainer`` class
-------------------------------------

The ``PyTorchParallelTrainer`` class
------------------------------------




