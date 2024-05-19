# RL2-implementation-pytorch
This is an implementation of the algorithmic idea presented in “Rl2: Fast reinforcement learning via slow reinforcement learning” and Learning to reinforcement learn”. The implementation applies it to the Meta World benchmark.


### SOME NOTES:

An important goal of this implementation is for it to be easy to understand. With that intention in mind, it is highly commented.

It is implemented for the Meta World benchmarks but it should be easy to adapt it to other gymnasium benchmarks. For this, the tasks initialization, the tasks sampler, the reward normalization and the end of episode handling should be modified.

The code contains non crucial components like logging with weights and biases, reward normalization and simple model saving.

Multiprocessing for parallel environment execution is achieved through Ray.

#### Example run:
Finally, a training run is shown on the Meta World ML10 benchmark. It attains a similar performance to that given in "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning".
![Screenshot of the project](_assets/ML10_run.png)
