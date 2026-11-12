"""Imageio and NumPy examples corresponding to the UW image section.

The input is a crop of the cat photo in the UW CSE163 RGB-layer figure.
https://courses.cs.washington.edu/courses/cse163/26su/lesson6-3/
Outputs are ordinary PNG files. No network, window or model weights are needed.
"""
from pathlib import Path
import imageio.v3 as iio
import numpy as np
np.set_printoptions(precision=2, floatmode="fixed", suppress=True)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "images" / "uw_cat.png"
OUT = HERE / "image-outputs"
OUT.mkdir(exist_ok=True)

# %% Read HWC, then use CHW for all image calculations
raw = iio.imread(SOURCE)
img = raw.transpose(2, 0, 1).copy()

def save_rgb(name, chw):
    """imageio expects HWC when writing an RGB file."""
    iio.imwrite(OUT / name, chw.transpose(1, 2, 0))

print("RGB shape, dtype:", img.shape, img.dtype)
print("Pixel [:,90,140]:", img[:, 90, 140])
print("Red [0,90,140]:", img[0, 90, 140])
print("One-channel shape:", img[0].shape)
assert img.shape == (3, 190, 280) and img.flags["C_CONTIGUOUS"]
np.testing.assert_array_equal(img[:, 90, 140], [131, 89, 64])
np.testing.assert_array_equal(img.transpose(1, 2, 0), raw)
save_rgb("copy.png", img)

# %% Grayscale uses two axes
gray = iio.imread(SOURCE, mode="L")
print("Grayscale shape and corner:", gray.shape, gray[0, 0])
assert gray.shape == (190, 280) and gray[0, 0] == 202
iio.imwrite(OUT / "gray.png", gray)
edited = gray.copy()
edited[-50:, :] = 100
iio.imwrite(OUT / "gray_band.png", edited)

# %% Slices modify regions and channels
edited = img.copy()
edited[0, 50:80, :] = 0
save_rgb("no_red_band.png", edited)
edited[1, 110:140, :] = 0
save_rgb("two_bands.png", edited)
edited[:, :, 80:100] = 255
save_rgb("bands_and_column.png", edited)

# %% Cropping and subsampling
crop = img[:, 30:150, 80:240].copy()
small = img[:, ::2, ::2]
print("Crop:", crop.shape, "Subsampled:", small.shape)
save_rgb("crop.png", crop)
save_rgb("small.png", small)

# %% One RGB rule broadcasts over all spatial positions
gain = np.array([1., 0.5, 1.])[:, None, None]
changed = img.astype(np.float32) * gain
changed = np.clip(changed, 0, 255).astype(np.uint8)
save_rgb("channel_gain.png", changed)
np.testing.assert_array_equal(changed[0], img[0])
np.testing.assert_array_equal(changed[2], img[2])

# %% Float arithmetic, clipping and conversion
bright = img.astype(np.float32) + 40
bright = np.clip(bright, 0, 255).astype(np.uint8)
save_rgb("bright.png", bright)
pixels = img.astype(np.float32) / 255
print("Float input range:", float(pixels.min()), float(pixels.max()))
channel_mean = np.array([.48145466,.4578275,.40821073],dtype=np.float32)
channel_std = np.array([.26862954,.26130258,.27577711],dtype=np.float32)
mean = channel_mean[:,None,None]
std = channel_std[:,None,None]
normalized = (pixels - mean) / std
assert normalized.shape == img.shape and normalized.dtype == np.float32
print("Normalized CHW shape, dtype:", normalized.shape, normalized.dtype)

# %% A slice shares storage; a copy is independent
working = img.copy()
view = working[:, 30:150, 80:240]
independent = working[:, 30:150, 80:240].copy()
before = independent.copy()
view[:] = 0
assert np.all(working[:, 30:150, 80:240] == 0)
np.testing.assert_array_equal(independent, before)

# %% A batch and a change of axis order
batch = np.stack([pixels, pixels], axis=0)
assert batch.shape == (2, 3, 190, 280)
print("BCHW:", batch.shape)
toy = np.ones((3, 2, 3))
print("CHW sum axis shapes:", [toy.sum(axis=i).shape for i in range(3)])
print("PNG outputs:", OUT)
