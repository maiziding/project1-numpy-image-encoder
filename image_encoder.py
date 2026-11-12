"""NumPy image encoder with supplied pretrained parameters.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """Split the feature axis into heads while preserving token order."""
        batch_size, token_count, _ = x.shape
        return x.reshape(batch_size, token_count, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # Transpose each key matrix to compare every query with every key.
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # Mix values within each head, then restore the original feature axis.
        # context: [B,H,N,64], merged: [B,N,D].
        context = coefficients @ v
        batch_size, token_count, _ = x.shape
        merged = context.transpose(0, 2, 1, 3).reshape(batch_size, token_count, self.width)

        # Apply the output projection.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # Add the attention update to the block input.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # Apply the MLP and add the second residual.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """Flatten 32x32 RGB patches in row-major grid order."""
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected image shape (B, 3, 224, 224), got {images.shape}")

        # Move the patch grid before C, H, W so each patch flattens in CHW order.
        batch_size = images.shape[0]
        grid = images.reshape(batch_size, 3, 7, 32, 7, 32)
        patches = grid.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch_size, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # Project the flattened patches with the bias-free patch weights.
        patch_tokens = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # Prepend CLS and broadcast positions across the batch.
        # Return [B,50,768]. self.position broadcasts over B.
        return np.concatenate((cls, patch_tokens), axis=1) + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """Return one unit-normalized 512-dimensional feature per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        cls_token = tokens[:, 0, :]
        output = self.post_norm(cls_token) @ self.projection

        return unit_vector(output)
