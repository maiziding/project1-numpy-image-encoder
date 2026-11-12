"""Run the lecture's miniature Transformer block, one calculation at a time.

Teaching example only: three token rows, four features, one attention head,
and six MLP middle values. These toy parameters are not the learned checkpoint.
The assessed model still needs the six TODO groups in image_encoder.py.
This example does not implement head splitting or merging.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights


def normalize_rows(x, weight, bias):
    # The feature axis runs left to right inside EACH token row.
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    standardized = centered / np.sqrt(variance + 1e-5)
    normalized = standardized * weight + bias
    np.testing.assert_allclose(normalized, LayerNorm(weight, bias)(x))
    return normalized, mean, centered, variance, standardized


def example_input():
    # Row 0 is a miniature CLS token, rows 1 and 2 are patch tokens.
    return np.array([[0.80, -0.60, 0.70, 0.30],
                     [-0.40, 0.90, -0.30, 0.50],
                     [0.10, -0.70, 0.60, -0.20]], dtype=np.float32)


def example_parameters():
    rng = np.random.default_rng(24)
    def values(shape):
        return rng.uniform(-0.65, 0.65, shape).round(2).astype(np.float32)
    return {
        'norm1_weight': np.array([0.90, 1.10, 1.00, 0.80], dtype=np.float32),
        'norm1_bias': np.array([0.10, -0.05, 0.02, -0.03], dtype=np.float32),
        'qkv_weight': values((12, 4)), 'qkv_bias': values((12,)),
        'out_weight': values((4, 4)), 'out_bias': values((4,)),
        'norm2_weight': np.array([1.00, 0.80, 1.10, 0.90], dtype=np.float32),
        'norm2_bias': np.array([-0.02, 0.03, 0.01, -0.04], dtype=np.float32),
        'fc1_weight': values((6, 4)), 'fc1_bias': values((6,)),
        'fc2_weight': values((4, 6)), 'fc2_bias': values((4,)),
    }


def walk_block(x, theta):
    # Norm 1 reads x. The residual path keeps the original x.
    n1, mean1, centered1, variance1, standardized1 = normalize_rows(
        x, theta['norm1_weight'], theta['norm1_bias'])

    # Stored linear weights are [out,in]. Bias repeats for every token row.
    qkv = n1 @ theta['qkv_weight'].T + theta['qkv_bias']
    q = qkv[:, :4]
    k = qkv[:, 4:8]
    v = qkv[:, 8:]

    # One teaching head, four features. The real model uses head_dim=64.
    scores = (q @ k.T) / np.sqrt(np.float32(4.0))
    coefficients = attention_weights(scores)  # supplied conversion helper
    context = coefficients @ v
    update = context @ theta['out_weight'].T + theta['out_bias']

    # First residual: add to x, not to the normalized n1.
    residual = x + update

    # Norm 2 reads residual. The second residual path keeps residual.
    u, mean2, centered2, variance2, standardized2 = normalize_rows(
        residual, theta['norm2_weight'], theta['norm2_bias'])
    hidden = u @ theta['fc1_weight'].T + theta['fc1_bias']
    activated = GELU()(hidden)  # supplied exact activation
    mlp_update = activated @ theta['fc2_weight'].T + theta['fc2_bias']
    output = residual + mlp_update

    trace = {key: value.copy() for key, value in locals().items()
             if isinstance(value, np.ndarray)}
    return output, trace


if __name__ == '__main__':
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    x, theta = example_input(), example_parameters()
    output, trace = walk_block(x, theta)
    stages = ('x', 'mean1', 'centered1', 'variance1', 'standardized1', 'n1',
              'qkv', 'q', 'k', 'v', 'scores', 'coefficients', 'context',
              'update', 'residual', 'u', 'hidden', 'activated',
              'mlp_update', 'output')
    for name in stages:
        value = trace[name]
        print(f'\n{name}: shape={value.shape}\n{value}')
    np.testing.assert_allclose(trace['coefficients'].sum(-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(trace['residual'], x + trace['update'])
    np.testing.assert_allclose(output, trace['residual'] + trace['mlp_update'])
    assert output.shape == x.shape == (3, 4)
    assert output.dtype == np.float32 and np.isfinite(output).all()
    print('\nMiniature forward complete. Display values round to two decimals.')
