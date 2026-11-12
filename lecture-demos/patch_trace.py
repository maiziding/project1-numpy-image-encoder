"""A tiny supplied input path: image -> patch rows -> tokens -> CLS/positions.

Uses toy parameters, not a trained model. Multi-head attention remains a student
TODO in the student image_encoder.py at the package root. This tiny example is only arithmetic,
not the actual model input, parameter layout or reference output.
"""
import numpy as np
np.set_printoptions(precision=2, floatmode="fixed", suppress=True)

# %% A normalized toy image: H=4, W=4; batch B=1, channel C=1
image = np.array([[-0.5, 0.89, -0.62, -0.64], [-0.3, -0.54, 0.34, -0.77], [0.79, 0.72, -0.99, 0.08], [-0.79, -0.48, -0.17, -0.09]], dtype=np.float32)
pixels = image[None, None, :, :]
B,C,H,W = pixels.shape
ps = 2
patches = []
for row in range(0, H, ps):
    for col in range(0, W, ps):
        patch = pixels[:, :, row:row+ps, col:col+ps]
        patches.append(patch.reshape(B, -1))
patches = np.stack(patches, axis=1)
print("BCHW:", pixels.shape, "patch rows:", patches.shape)
print(patches[0])
np.testing.assert_allclose(patches[0,0], [-0.50,0.89,-0.30,-0.54])
assert patches.shape == (1,4,4)

# %% Each flattened patch gets D=3 token values through matrix multiplication
rng = np.random.default_rng(7)
patch_w = rng.normal(0, .2, (C*ps*ps, 3)).astype(np.float32)
patch_b = np.zeros(3, dtype=np.float32)
tokens = patches @ patch_w + patch_b
print("Patch tokens:", tokens.shape, "\n", tokens[0])
assert tokens.shape == (1,4,3)

# %% CLS is a supplied learned row; positions are supplied learned values
cls_token = np.array([[[.11,.28,.39]]], dtype=np.float32)
cls = np.tile(cls_token, (B,1,1))
x = np.concatenate([cls, tokens], axis=1)
positions = rng.normal(0, .01, x.shape).astype(np.float32)
x = x + positions
print("CLS + four patches + positions:", x.shape)
assert x.shape == (1,5,3)
print("Next: Transformer blocks preserve (B,L,D).")
print("After all blocks: select CLS x[:,0,:], then apply post-Norm and projection.")
