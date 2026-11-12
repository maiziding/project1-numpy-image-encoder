"""Provided asset loading. Paths are relative to this package, not the shell cwd."""
from pathlib import Path
import json
import numpy as np
from preprocess_image import PreprocessImage

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_weights():
    config = read_json(ROOT/'config.json')
    with np.load(ROOT/config['weights'], allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def image_records():
    return read_json(ROOT/'assets/images.json')


def load_images(paths=None):
    paths = paths or [ROOT/'assets'/r['path'] for r in image_records()]
    prepare = PreprocessImage()
    return np.concatenate([prepare(Path(p)) for p in paths], axis=0)
