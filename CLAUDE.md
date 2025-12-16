# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A JAX program to train a neural network to recognize an XOR function. The program should support three modes of operation:
1. Generate random training/test data given a specified seed
2. Train and save network weights
3. Run a saved network through test data and report error rate

## XOR Function Specification

- Inputs and outputs are in range 0 to 1.0
- Threshold of 0.5 determines boolean value (≥0.5 = ON, <0.5 = OFF)
- Output should be 1.0 (ON) or 0.0 (OFF) matching XOR logic
- Example: func(0.8, 0.7) → 0.0, func(0.3, 0.9) → 1.0

## Network Architecture

Use a simple two-layer network: one hidden layer with a small number of neurons plus an output layer. XOR is solvable with minimal architecture and trains quickly with this approach.
