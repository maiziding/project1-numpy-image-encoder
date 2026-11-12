NUMPY CLASSROOM DEMONSTRATIONS

Install the package requirements from the parent directory:
  python -m pip install -r requirements.txt

Run from the package root:
  python lecture-demos/01_numpy_basics.py
  python lecture-demos/02_imageio_images.py
  python lecture-demos/03_forward_tokens.py
  python lecture-demos/04_join_arrays.py
  python lecture-demos/patch_trace.py
  python transformer_steps.py

01: arrays, shapes, selections, products, reductions and broadcasting.
02: local cat image, HWC/CHW, safe arithmetic, slices and image batches.
03: toy token lookup and small forward calculations, not learned model answers.
04: concatenate/stack and image batches.
patch_trace: toy pixels, patch rows, projection and CLS/positions.
transformer_steps.py: the supplied miniature block companion at the package root.

The image example writes ordinary PNGs to lecture-demos/image-outputs/.
expected-outputs/ records the small numerical demonstration results.
The assessed learned image encoder is the incomplete image_encoder.py in the
student edition. The demonstrations do not replace its weights or TODOs.
No network or model weights are required by these classroom demonstrations.
