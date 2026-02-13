# Learn a Quadratic Function

The goal of this module is to learn how to build a simple neural network that is
capable of modeling the behavior of a quadratic function.

The data we have for training is in train.dat, and the data for testing is in
test.dat.

There are two variants of the model we train. One variant uses the torchlite
module, which is similar to PyTorch, but instead of being optimized for
performance, is optimized for simplicity so that we can easily trace how call
stacks work. The second version is written with PyTorch.

This allows us to train both variants and compare their performance.
