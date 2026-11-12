"""Run cell by cell in VS Code: joins preserve existing axes or add a new one."""
from pathlib import Path
import imageio.v3 as iio
import numpy as np
np.set_printoptions(precision=2, floatmode="fixed", suppress=True)

# %% Join on an existing axis: add rows or columns
a = np.array([[0.80,-0.60],[-0.40,-0.50]])
b = np.array([[.7, .9]])
rows = np.concatenate([a, b], axis=0)
columns = np.concatenate([a, np.array([[.7], [.9]])], axis=1)
print("A:\n", a)
print("More rows:\n", rows, rows.shape)
print("More columns:\n", columns, columns.shape)
assert rows.shape == (3, 2) and columns.shape == (2, 3)
np.testing.assert_array_equal(rows[:2], a)

# %% New batch axis: two CHW images become BCHW
raw = iio.imread(Path(__file__).resolve().parent / "images" / "uw_cat.png")
image_a = raw.transpose(2, 0, 1).astype(np.float32) / 255
image_b = image_a[:, :, ::-1].copy()
batch = np.stack([image_a, image_b], axis=0)
print("One image CHW:", image_a.shape)
print("Two images BCHW:", batch.shape)
assert batch.shape == (2, 3, 190, 280)
np.testing.assert_array_equal(batch[1], image_b)

# %% Join existing batches: B grows; C/H/W stay unchanged
first = batch
second = image_a[None, ...]
combined = np.concatenate([first, second], axis=0)
print("Existing batches:", first.shape, "+", second.shape, "=", combined.shape)
assert combined.shape == (3, 3, 190, 280)

# %% Same inputs, different meaning: concatenation grows C, not B
channel_join = np.concatenate([image_a, image_b], axis=0)
print("Channel join (not a two-image batch):", channel_join.shape)
assert channel_join.shape == (6, 190, 280)
try:
    np.stack([image_a, image_b[:, :-1, :]], axis=0)
except ValueError:
    print("stack needs equal input shapes: resize/crop before stacking.")
else:
    raise AssertionError("Unequal image shapes should fail")
