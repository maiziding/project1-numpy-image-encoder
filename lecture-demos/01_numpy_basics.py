"""Signed NumPy examples for the revised lecture. Run # %% cells in VS Code.

Floating values display with two decimals. Indices, shape lengths and IDs remain
integers. Calculations retain full precision. Images use CHW/BCHW/BTCHW.
"""
import numpy as np
np.set_printoptions(precision=2, floatmode="fixed", suppress=True)

# %% Startup and reproducible signed random arrays
a = np.array([0.80, -0.60, 0.70])
print("array offset:", a + 0.10)
rng = np.random.default_rng(7)
print("uniform(-1,1):", rng.uniform(-1, 1, (2, 3)))
print("zeros:", np.zeros((2, 3)))
print("ones:", np.ones((2, 3)))
print("linspace:", np.linspace(-0.80, 0.80, 5))
print("normal:", rng.normal(size=(2, 3)))
# Reuse one signed sample throughout the arithmetic diagrams.
v = np.random.default_rng(7).choice(np.arange(1, 10), 6, replace=False) / 10
v = v * np.array([1, -1, 1, -1, -1, 1])
np.testing.assert_allclose(v, [0.80, -0.60, 0.70, -0.40, -0.50, 0.90])
a = v.reshape(2, 3)

# %% 0D to 5D: one value, sequence, grid, image, image batch, video batch
rng = np.random.default_rng(7)
scalar = np.array(-0.40)
image = rng.uniform(-1, 1, (3, 2, 3))  # normalized toy image values
batch = rng.uniform(-1, 1, (2, 3, 2, 3))
video = rng.uniform(-1, 1, (2, 3, 3, 2, 2))
for name, value in [("scalar", scalar), ("sequence", v), ("grid", a),
                    ("CHW image", image), ("BCHW batch", batch),
                    ("BTCHW video batch", video)]:
    print(name, "shape:", value.shape, "ndim:", value.ndim, "size:", value.size)
assert video.ndim == 5 and batch.ndim == 4
print("normalized image[0,0,0]:", image[0,0,0].round(2))

# %% Select and rearrange
print("row 1:", a[1])
print("column range:", a[:,1:3])
print("new row axis:", a[1:2].shape)
print("reshape:", v.reshape(3, 2))
print("transpose:", a.T)
x = v[:3]
print("flat, row, column:", x.shape, x[None,:].shape, x[:,None].shape)
mask = v > 0.65
print("mask:", mask, "selected:", v[mask])

# %% A single rule applies to every value
loop = [value*0.50+0.10 for value in x]
print("vectorized rule:", x*0.50+0.10)
np.testing.assert_allclose(x*0.50+0.10, loop)
print("absolute values:", np.abs(x), "squares:", x**2)

# %% Multiplication: elementwise, dot, vector@matrix, matrix@matrix
left, right = v[:3], v[3:]
W = np.array([[-0.10,0.30], [0.20,-0.50], [-0.80,0.10]])
print("products:", left*right)
print("dot:", (left@right).round(2))
print("W:", W)
print("vector @ matrix:", left@W)
print("matrix @ matrix:", a@W)
np.testing.assert_allclose(left*right, [-0.32,0.30,0.63])
np.testing.assert_allclose(left@right, 0.61)
np.testing.assert_allclose(left@W, [-0.76,0.61])
np.testing.assert_allclose(a@W, [[-0.76,0.61],[-0.78,0.22]])
u = np.array([[0.80,-0.60],[-0.40,-0.50]])
z = np.array([[0.20,-0.30],[0.70,0.90]])
print("2D elementwise:", u*z, "2D matrix product:", u@z)
np.testing.assert_allclose(u@z, [[-0.26,-0.78],[-0.43,-0.33]])

# %% Reductions and broadcasting
print("C-order sequence:", a.ravel())
print("sum axis 0:", a.sum(0), "sum axis 1:", a.sum(1))
np.testing.assert_allclose(a.sum(0), [0.40,-1.10,1.60])
np.testing.assert_allclose(a.sum(1), [0.90,0.00], atol=1e-15)
print("row offsets:", a+[0.10,-0.20,0.30])
print("column offsets:", a+[[0.10],[-0.20]])
mean = a.mean(axis=1, keepdims=True)
print("row means:", mean, "centered rows:", a-mean)
np.testing.assert_allclose(a-mean, [[0.50,-0.90,0.40],[-0.40,-0.50,0.90]])
for axis in range(3):
    print("CHW reduction axis", axis, "shape:", image.sum(axis).shape)
print("All signed array examples passed.")
