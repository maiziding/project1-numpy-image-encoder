"""Signed token rows and forward calculations; helper internals are supplied."""
import numpy as np
def attention_weights(scores):
    """Provided stable conversion; the class focuses on weighted sums."""
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)
np.set_printoptions(precision=2, floatmode="fixed", suppress=True)

# %% Token IDs address rows, rather than being numerical meanings
E = np.random.default_rng(8).uniform(-0.90,0.90,(4,4)).round(2)
words = "cat dog cat".split()
vocab = {"cat":1, "dog":2, "boat":3}
ids = np.array([vocab[word] for word in words])
print("token table:", E, "IDs:", ids, "selected rows:", E[ids])
np.testing.assert_array_equal(E[ids][0], E[ids][2])
batch_ids = np.array([[1,2,1],[3,1,3]])
print("BLD:", E[batch_ids].shape)
x = np.array([0.80,-0.60,0.70])
position = np.array([0.04,-0.02,0.09])
print("add positions:", x+position)

# %% Two matrix products and an elementwise activation
W1 = np.array([[-0.10,0.30],[0.20,-0.50],[-0.80,0.10]])
W2 = np.array([[0.20,-0.80],[-0.60,0.52]])
b2 = np.array([0.00,0.00])  # supplied output bias for the toy MLP
sigma = lambda z: np.maximum(z,0.00)
simple_y = sigma(x@W1)@W2
print("no-bias hidden:", sigma(x@W1), "output:", simple_y)
np.testing.assert_allclose(simple_y, [-0.366,0.3172])
b1 = np.array([-0.15,0.05])
theta = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
# These model parameters stay fixed while the input token x changes.
z = x@W1+b1
h = sigma(z)
y = h@W2+b2
print("linear:", x@W1, "bias added:", z, "ReLU:", h, "output:", y)
np.testing.assert_allclose(y, [-0.396,0.3432])
X = np.array([[0.80,-0.60,0.70],[-0.40,-0.50,0.90]])
Z = X@W1+b1
Y = sigma(Z)@W2+b2
print("batch Z:", Z, "batch Y:", Y)
np.testing.assert_allclose(Y, [[-0.396,0.3432],[-0.162,0.1404]])

# %% Unit vectors and signed similarity scores
q = np.array([-0.40,0.30])
q = q/np.sqrt((q*q).sum())
G = np.array([[0.80,0.60],[-0.60,0.80]])
scores = G@q
print("query:", q, "scores:", scores, "ranked IDs:", np.argsort(scores)[::-1])
np.testing.assert_allclose(scores, [-0.28,0.96])

# %% A normalized toy image, then row-major patches
img = np.random.default_rng(12).uniform(-1,1,(4,4)).round(2)
patches = np.stack([img[r:r+2,c:c+2].reshape(-1)
                    for r in range(0,4,2) for c in range(0,4,2)])
fast = img.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
print("normalized toy image:", img, "patches:", patches)
np.testing.assert_array_equal(fast, patches)
np.testing.assert_allclose(patches[0], [-0.50,0.89,-0.30,-0.54])

# %% LayerNorm operates inside each token row
row = x
mean = row.mean(-1,keepdims=True)
var = row.var(-1,keepdims=True)
print("row mean:", mean, "centered row:", row-mean,
      "normalized row:", (row-mean)/np.sqrt(var+1e-6))

# %% Scores -> provided attention weights -> weighted row mixture
q = x
k, v = E[1:4,:3], E[:3,:3]
raw_scores = q@k.T
weights = attention_weights(raw_scores/np.sqrt(3))
result = weights@v
print("query:", q, "keys:", k)
print("raw scores:", raw_scores, "weights:", weights, "output:", result)
np.testing.assert_allclose(raw_scores, [0.579,-0.873,0.266])
np.testing.assert_allclose(weights.sum(),1.00)
np.testing.assert_allclose(result, [-0.2703372478934684,0.33528273329170133,-0.33957958936501104], atol=1e-8)
print("All forward examples passed.")
