"""Provided command line: images -> your ImageEncoder -> output tokens."""
import argparse
from pathlib import Path
import numpy as np
from runtime import load_images, load_weights, image_records
from image_encoder import ImageEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', help='Image paths. Default: ten supplied images.')
    parser.add_argument('--output', type=Path, default=Path('output_tokens.npz'))
    args = parser.parse_args()
    try:
        encoder = ImageEncoder(load_weights())
        images = load_images(args.images)
        result = encoder(images)
    except NotImplementedError as exc:
        parser.exit(2, 'Implementation incomplete: '+str(exc)+'\n')
    if result.shape != (len(images),512) or result.dtype != np.float32 or not np.isfinite(result).all():
        parser.exit(1, 'Expected finite float32 output with shape [B,512].\n')
    ids = [Path(p).stem for p in args.images] if args.images else [r['id'] for r in image_records()]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), output_tokens=result)
    print('Saved', result.shape, 'to', args.output)


if __name__ == '__main__':
    main()
