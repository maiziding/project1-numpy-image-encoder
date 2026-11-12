# Project 1: reproduce the final project's image encoder

Given images and learned model parameters **theta**, implement the OpenCLIP
ViT-B-32 **forward calculation in NumPy**. Your output must numerically match
the supplied reference. The same completed encoder becomes the image tower of
`szu_python_openclip` in the final project. No model learning is required.

## Input and output

- Input to `ImageEncoder`: float32 array **[B,3,224,224]** after the supplied preprocessing.
- Internal image-token array after 12 blocks: **[B,50,768]** (CLS plus 49 patch tokens).
- Final **output token**: float32 **[B,512]**, one normalized numerical row per image.
  This is exactly the output of the final project's `ImageEncoder.__call__`.

The term output token in this assignment means the final 512-value row.
`last_tokens` separately names the complete 50-row internal sequence.


## Files in this attachment

Unzip the attachment and open this directory. Commands in this README run from
this directory; Python resolves model/image files relative to the scripts.
No absolute course-directory path is required.

- `practice/project1_exercises.pdf`: 12 NumPy practice questions, without answers.
- `lecture-demos/`: the NumPy classroom examples and the credited local cat image.
- `transformer_steps.py`: the supplied miniature block companion.
- `PPT-CODE-MAP.txt`: the current 76-slide lecture map.
- `SOURCE-MANIFEST.json`: provenance, SHA-256 and intentional packaging changes.

The assignment needs NumPy and Pillow. The imageio classroom example also needs
imageio; the attachment requirements include all three.

## Quick start

Use Python 3.9 or later. No GPU, PyTorch, OpenCLIP package or network is needed.

```bash
python -m pip install -r requirements.txt
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
```

The first two checks pass in the untouched starter. Later checks report the
first missing TODO instead of pretending the starter is complete. After filling
the six exercises:

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

Run scripts from any directory by using their full paths if needed. `run.py`
saves image IDs and output tokens in the same order as its inputs.

## Follow the lecture's calculation

```bash
python transformer_steps.py
```

This supplied companion illustrates the arithmetic explained on slides 46-66:
row means, centering, variance, Norm scale/bias, q/k/v, scores, coefficients,
weighted values, attention projection, both residuals and the MLP. The lecture
uses full image-array shapes; the companion prints a miniature calculation.
The example uses **three token rows, four features, one head and six MLP middle
values**. Its synthetic parameters keep all values visible. It is not the
learned checkpoint. Display values round to two decimals; calculations retain
full float32 precision.

Transfer the arithmetic to the real shapes and TODOs in `image_encoder.py`.
The Norm 2 result is named `u`, matching the assessed encoder.
The miniature uses a 2D table, so `k.T` swaps its two axes. Real k is 4D:
exchange only its final two axes and preserve image/head groups. Study and
implement head organization independently. The companion leaves that part out.
The real model still uses the exact supplied image parameters and reference
outputs. All six assessed exercise groups remain incomplete.

Slides 67-70 separate the final output into CLS selection, CLS post-Norm,
output projection and unit-length scaling. Slides 71-73 collect the encoder
contract, six implementation tasks, classroom commands and numerical checks.
Slides 74-76 explain final-project reuse and the text extension.
The current lecture contains 76 slides.

## What to implement

Edit **image_encoder.py** only. A single exercise can have several marked steps.

| Exercise | Method | Required result |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | [B,3,224,224] to [B,49,3072], patch order preserved |
| TODO-02 | `MultiHeadAttention.split_heads` | [B,N,D] to [B,H,N,D/H] |
| TODO-03 | `MultiHeadAttention.__call__` | scores, weighted values, head merge and output projection |
| TODO-04 | `TransformerBlock.__call__` | attention/MLP calculations and both residual additions |
| TODO-05 | `ImageEncoder.embed` | patch projection, CLS concatenation and learned positions |
| TODO-06 | `ImageEncoder.__call__` | select CLS, post-Norm, projection and final output |

Use NumPy array operations, indexing, reshape/transpose, matrix multiplication,
reductions and broadcasting. Learn the multi-head organization independently.
The activation and attention-coefficient conversion are already provided.

## What is provided

- `preprocess_image.py`: exact resize, center crop and pixel normalization.
- `helpers.py`: LayerNorm, exact GELU, attention coefficients and vector scaling.
- `runtime.py`, `run.py`, `check.py`: loading, command line and reference checks.
- `weights/image_encoder.npz`: **only the final model's 152 visual tensors**.
  Original tensor names, shapes and float32 values are preserved. No text weights.
- `config.json` and `weights/manifest.json`: exact architecture and fingerprints.
- `assets/images/`, `images.json`: ten local images with stable IDs and provenance.
- `assets/reference_outputs.npz`: independent FP32 stage/output arrays in ID order.

The config describes this fixed checkpoint. Changing a config number does not
turn it into another model. Follow the supplied weights and interfaces.

## Numerical checks

`check.py` compares against an independent Torch FP32 forward and the final
project's teacher reference. Students use only NumPy. Matching means equal shape,
finite float32 values and `abs(actual-reference) <= atol + rtol*abs(reference)`.
Output-token tolerance is **atol=2e-5, rtol=2e-4**. Internal token tolerance is
**atol=2e-4, rtol=2e-4**. Preprocessing uses atol=2e-6. Checks print maximum and
mean absolute error. `batch` checks changed order and duplicate images.

Useful commands, in implementation order:

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

Inspect the targets without changing the provided files:

```python
import numpy as np
with np.load("assets/reference_outputs.npz", allow_pickle=False) as ref:
    print(ref["ids"])
    print(ref["last_tokens"].shape)    # (10,50,768)
    print(ref["output_tokens"].shape)  # (10,512)
```

`PPT-CODE-MAP.txt` maps the lecture excerpts to these exact starter lines.

Read the intermediate arrays to locate the **first** stage that differs. They
are debugging targets, never inputs to your encoder. Your implementation must
also handle other images and batch sizes. Do not return cached reference rows.

## Important model conventions

1. Patches are 32x32, so 224/32 gives 7x7 patches. Each patch flattens C,H,W.
2. Patch projection has **no bias**. It uses `patch_weight.T`.
3. Attention/MLP linear weights are stored **[out,in]**, so use `weight.T`.
4. `visual.proj` is already **[in,out] = [768,512]**. Do not transpose it.
5. Use the supplied exact GELU and Norm epsilon. No parameter updates occur.
6. The image block has two residual additions. It has no LayerScale parameter.

## Submission

Submit your completed `image_encoder.py`, a short README explaining how to run it,
and your report. Include per-stage numerical errors and explain one shape change
and one bug you fixed. Include your contributions and assistance used. Do not
resubmit the large supplied parameter file or reference data. Assessment follows
the course's code 30 / report 70 rule and announced interview/deadline policy.

## Reuse in the final project

Copy completed `image_encoder.py` **and the provided `helpers.py`** beside `clip.py` and `text_encoder.py` in the separately distributed final-project
attachment. Paths are relative to that extracted attachment. `ImageEncoder(weights)`,
`LayerNorm` and `TransformerBlock(weights,prefix,heads)` keep the required APIs.
The provided optional causal switch preserves later text-tower compatibility.
The final project still supplies the full image/text checkpoint and tokenizer.
Do not substitute these image-only weights for the full multimodal checkpoint.
