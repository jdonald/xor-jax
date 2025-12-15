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

## Network Architecture

Simple 2-layer network: 2 inputs → 4 hidden neurons (sigmoid) → 1 output (sigmoid).
Uses BCE loss and SGD optimizer.

## Requirements

```
python3 -m virtualenv myenv
myenv/bin/activate
pip install -r requirements.txt
```
(but actually the only dep in `requirements.txt` is `pip install torch`)

## Usage

```bash
# Generate data
python3 xor_net.py generate --seed 42 --samples 1000 --output train_data.json

# Train
python3 xor_net.py train --data train_data.json --weights weights.pt --epochs 10000 --lr 2.0

# Test (prints GPU then CPU benchmarks)
python3 xor_net.py test --data test_data.json --weights weights.pt
```

## Testing Inference

When testing inference, print benchmarks from running first on the GPU and then
on the CPU.

Sample output:
```
=== GPU Benchmark ===
Device: Tesla T4
Inference time: 0.1907 ms
Samples: 200
Correct: 196/200
Error rate: 2.00%

=== CPU Benchmark ===
Inference time: 0.0703 ms
Samples: 200
Correct: 196/200
Error rate: 2.00%
```

Yeah, the CPU run is consistently three times faster than the GPU one. Similar results on an Ada Lovelace mobile GPU.
Perhaps that's the way it is with small two-layer networks.
