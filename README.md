# xor-pytorch

Vibe-coded PyTorch exploration to train a network to recognize an XOR function

## Description

A small program using PyTorch to train a neural network to recognize an XOR function.
This Python program should have options to do any of:
1) Generate random training/test data given a specified seed
2) Train and save network weights
3) Run a saved network through set of test data and report its error rate.

XOR function for our intents and purposes:
Inputs and outputs of the network are in the range 0 to 1.0. Treat a threshold
as 0.5 or higher to mean a boolean ON, below that to be OFF. Ideal output is
then 1.0 (ON) or 0.0 (OFF) matching what XOR would do. e.g. func(0.8, 0.7) --> 0.0
func(0.3, 0.9) --> 1.0
