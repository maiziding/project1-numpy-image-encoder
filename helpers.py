"""Provided numerical helpers. No student edits are required."""
import math
import numpy as np

class LayerNorm:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        """Provided: Normalize each token's features; input/output [B,L,D]."""
        # Step 1: Average the last axis and retain it for broadcasting; [B,L,1].
        mean = x.mean(axis=-1, keepdims=True)

        # Step 2: Subtract the mean from each feature; [B,L,D].
        centered = x - mean

        # Step 3: Average squared deviations to obtain the variance; [B,L,1].
        variance = (centered * centered).mean(axis=-1, keepdims=True)

        # Step 4: Divide by sqrt(variance + 1e-5) to avoid division by zero.
        normalized = centered / np.sqrt(variance + 1e-5)

        # Step 5: Apply the learned scale and bias by broadcasting; [B,L,D].
        return normalized * self.weight + self.bias

class GELU:
    def __call__(self, x):
        """Provided: Exact OpenCLIP GELU using basic loops and standard-library erf."""
        # Step 1: Flatten the array to visit scalar values with a plain Python loop.
        values = x.reshape(-1)
        activated = []

        # Step 2: Evaluate erf(x/sqrt(2)); math.erf is a Python standard-library function.
        divisor = math.sqrt(2.0)
        for value in values:
            activated.append(math.erf(float(value) / divisor))

        # Step 3: Restore the original shape and float32 dtype.
        erf_values = np.array(activated, dtype='float32').reshape(x.shape)

        # Step 4: Apply GELU(x)=0.5*x*(1+erf(x/sqrt(2))), matching the actual weights.
        return 0.5 * x * (1.0 + erf_values)

def attention_weights(scores):
    """Provided conversion from scores to nonnegative row coefficients."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def causal_mask(scores):
    """Provided for compatibility with the later text encoder. Images do not use it."""
    scores = scores.copy()
    for query in range(scores.shape[-2]):
        scores[..., query, query + 1:] = -1e9
    return scores


def unit_vector(x):
    """Provided final vector scaling. The assignment still selects/projects CLS."""
    length = np.sqrt((x * x).sum(axis=-1, keepdims=True))
    return x / np.maximum(length, np.float32(1e-12))
