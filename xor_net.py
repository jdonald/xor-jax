#!/usr/bin/env python3
"""
XOR Neural Network - Train a network to recognize XOR function using JAX.
"""

import argparse
import json
import time
import random
import pickle
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


class XORNet(nn.Module):
    """Simple 2-layer neural network for XOR function."""
    hidden_size: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        return x


def xor_label(a: float, b: float) -> float:
    """Compute XOR label for two float inputs using 0.5 threshold."""
    a_bool = a >= 0.5
    b_bool = b >= 0.5
    return 1.0 if a_bool != b_bool else 0.0


def generate_data(num_samples: int, seed: int) -> list[dict]:
    """Generate random training/test data for XOR function."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        a = random.random()
        b = random.random()
        label = xor_label(a, b)
        data.append({"inputs": [a, b], "label": label})
    return data


def save_data(data: list[dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filepath}")


def load_data(filepath: str) -> list[dict]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def data_to_arrays(data: list[dict]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert data list to JAX arrays."""
    inputs = jnp.array([[d["inputs"][0], d["inputs"][1]] for d in data], dtype=jnp.float32)
    labels = jnp.array([[d["label"]] for d in data], dtype=jnp.float32)
    return inputs, labels


def bce_loss(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss."""
    epsilon = 1e-7
    predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
    return -jnp.mean(labels * jnp.log(predictions) + (1 - labels) * jnp.log(1 - predictions))


def train(model: XORNet, data_path: str, weights_path: str, epochs: int = 1000, lr: float = 1.0, device: str = None):
    """Train the model and save weights."""
    if device is None:
        # Check if GPU is available
        devices = jax.devices()
        device_type = devices[0].platform
        print(f"Using device: {device_type}")

    # Load data
    data = load_data(data_path)
    inputs, labels = data_to_arrays(data)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 2)))

    # Setup optimizer
    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Define training step
    @jax.jit
    def train_step(params, opt_state, inputs, labels):
        def loss_fn(params):
            predictions = model.apply(params, inputs)
            return bce_loss(predictions, labels)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print(f"Training with {len(data)} samples for {epochs} epochs...")

    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, inputs, labels)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {float(loss):.6f}")

    # Save weights
    with open(weights_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved weights to {weights_path}")


def test(model: XORNet, data_path: str, weights_path: str):
    """Test the model and report error rate with GPU/CPU benchmarks."""
    data = load_data(data_path)
    inputs, labels = data_to_arrays(data)

    # Load weights
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)

    # Create JIT-compiled inference function
    @jax.jit
    def predict(params, inputs):
        return model.apply(params, inputs)

    # Test on GPU first (if available)
    devices = jax.devices()
    has_gpu = any(d.platform == 'gpu' for d in devices)

    if has_gpu:
        # Move data to GPU
        gpu_device = [d for d in devices if d.platform == 'gpu'][0]
        inputs_gpu = jax.device_put(inputs, gpu_device)
        params_gpu = jax.device_put(params, gpu_device)

        # Warmup
        for _ in range(10):
            _ = predict(params_gpu, inputs_gpu)

        # Block until all computations are done
        jax.block_until_ready(predict(params_gpu, inputs_gpu))

        start_time = time.perf_counter()
        outputs = predict(params_gpu, inputs_gpu)
        jax.block_until_ready(outputs)
        gpu_time = time.perf_counter() - start_time

        predictions = (outputs >= 0.5).astype(jnp.float32)
        correct = (predictions == labels).sum()
        error_rate = 1.0 - (correct / len(data))

        print(f"\n=== GPU Benchmark ===")
        print(f"Device: {gpu_device.device_kind}")
        print(f"Inference time: {gpu_time * 1000:.4f} ms")
        print(f"Samples: {len(data)}")
        print(f"Correct: {int(correct)}/{len(data)}")
        print(f"Error rate: {error_rate * 100:.2f}%")
    else:
        print("\nGPU not available, skipping GPU benchmark.")

    # Test on CPU
    cpu_device = [d for d in devices if d.platform == 'cpu'][0]
    inputs_cpu = jax.device_put(inputs, cpu_device)
    params_cpu = jax.device_put(params, cpu_device)

    # Warmup
    for _ in range(10):
        _ = predict(params_cpu, inputs_cpu)

    jax.block_until_ready(predict(params_cpu, inputs_cpu))

    start_time = time.perf_counter()
    outputs = predict(params_cpu, inputs_cpu)
    jax.block_until_ready(outputs)
    cpu_time = time.perf_counter() - start_time

    predictions = (outputs >= 0.5).astype(jnp.float32)
    correct = (predictions == labels).sum()
    error_rate = 1.0 - (correct / len(data))

    print(f"\n=== CPU Benchmark ===")
    print(f"Inference time: {cpu_time * 1000:.4f} ms")
    print(f"Samples: {len(data)}")
    print(f"Correct: {int(correct)}/{len(data)}")
    print(f"Error rate: {error_rate * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="XOR Neural Network - Train and test a network to recognize XOR function")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate random training/test data")
    gen_parser.add_argument("--seed", type=int, required=True, help="Random seed for data generation")
    gen_parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    gen_parser.add_argument("--output", type=str, default="data.json", help="Output file path (default: data.json)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the network and save weights")
    train_parser.add_argument("--data", type=str, default="data.json", help="Training data file (default: data.json)")
    train_parser.add_argument("--weights", type=str, default="weights.pkl", help="Output weights file (default: weights.pkl)")
    train_parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    train_parser.add_argument("--lr", type=float, default=1.0, help="Learning rate (default: 1.0)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the network and report error rate")
    test_parser.add_argument("--data", type=str, default="data.json", help="Test data file (default: data.json)")
    test_parser.add_argument("--weights", type=str, default="weights.pkl", help="Weights file to load (default: weights.pkl)")

    args = parser.parse_args()

    if args.command == "generate":
        data = generate_data(args.samples, args.seed)
        save_data(data, args.output)
    elif args.command == "train":
        model = XORNet()
        train(model, args.data, args.weights, args.epochs, args.lr)
    elif args.command == "test":
        model = XORNet()
        test(model, args.data, args.weights)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
