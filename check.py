"""Provided staged numerical checks. Reference arrays are used only here."""
from pathlib import Path
import argparse, hashlib, json, ast, sys
import numpy as np
from runtime import ROOT, read_json, load_weights, load_images, image_records

STAGES = ('assets','preprocess','patches','tokens','block0','last_tokens','output','batch')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unfinished():
    tree = ast.parse((ROOT/'image_encoder.py').read_text(encoding='utf-8'))
    return [n.lineno for n in ast.walk(tree) if isinstance(n,ast.Raise)
            and isinstance(n.exc,ast.Call) and isinstance(n.exc.func,ast.Name)
            and n.exc.func.id=='NotImplementedError']


def verify_assets():
    cfg = read_json(ROOT/'config.json')
    manifest = read_json(ROOT/'weights/manifest.json')
    assert digest(ROOT/cfg['weights']) == manifest['weights_sha256'], 'Wrong parameter file'
    assert manifest['tensor_count'] == 152
    weights = load_weights()
    assert set(weights) == set(manifest['tensors']), 'Wrong tensor names'
    for name, expected in manifest['tensors'].items():
        value = weights[name]
        assert list(value.shape)==expected['shape'] and str(value.dtype)==expected['dtype'], name
        assert hashlib.sha256(value.tobytes()).hexdigest()==expected['sha256'], name
    contract = {'image_size':224,'patch_size':32,'width':768,'heads':12,'layers':12,
                'token_count':50,'mlp_width':3072,'output_dim':512}
    assert all(cfg[k]==v for k,v in contract.items()), 'Config does not match this checkpoint'
    records = image_records()
    for record in records:
        assert digest(ROOT/'assets'/record['path']) == record['sha256'], record['id']
    expected = read_json(ROOT/'assets/manifest.json')
    assert digest(ROOT/'assets/reference_outputs.npz')==expected['reference_sha256']
    with np.load(ROOT/'assets/reference_outputs.npz',allow_pickle=False) as ref:
        assert ref['ids'].tolist()==[r['id'] for r in records], 'Image/reference order differs'
        assert ref['output_tokens'].shape==(10,512)
        assert ref['last_tokens'].shape==(10,50,768)
    return weights


def compare(stage, actual, expected):
    assert isinstance(actual,np.ndarray), stage+': return a NumPy array'
    assert actual.shape==expected.shape, f'{stage}: {actual.shape} != {expected.shape}'
    assert actual.dtype==np.float32, stage+': use float32'
    assert np.isfinite(actual).all(), stage+': found NaN or infinity'
    atol = 2e-6 if stage=='preprocess' else 2e-5 if stage in ('output','batch') else 2e-4
    rtol = 2e-4
    error = np.abs(actual-expected)
    ok = np.allclose(actual,expected,rtol=rtol,atol=atol)
    print(f'{stage:12} {"PASS" if ok else "FAIL"} shape={actual.shape} '
          f'max_abs={error.max():.3e} mean_abs={error.mean():.3e} '
          f'(atol={atol:g}, rtol={rtol:g})')
    assert ok, stage+': values differ from the independent FP32 reference'
    if stage in ('output','batch'):
        np.testing.assert_allclose((actual*actual).sum(-1),1.0,rtol=2e-4,atol=2e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all',)+STAGES,default='all')
    args = parser.parse_args()
    try:
        weights = verify_assets()
    except (AssertionError,ValueError,OSError,KeyError) as exc:
        parser.exit(1, 'assets FAIL: '+str(exc)+'\n')
    print('assets       PASS: 152 matching visual tensors and ten image/reference pairs')
    if args.stage=='assets':
        return 0
    from image_encoder import ImageEncoder
    encoder = ImageEncoder(weights)
    images = load_images()
    results = []
    with np.load(ROOT/'assets/reference_outputs.npz',allow_pickle=False) as ref:
        def compute(stage):
            if stage=='preprocess':return images,ref['preprocessed']
            if stage=='patches':return encoder.patchify(images[:2]),ref['patches'][:2]
            if stage=='tokens':return encoder.embed(images[:2]),ref['input_tokens'][:2]
            if stage=='block0':return encoder.blocks[0](encoder.pre_norm(encoder.embed(images[:2]))),ref['block0_tokens'][:2]
            if stage=='last_tokens':return encoder.encode_tokens(images[:2]),ref['last_tokens'][:2]
            if stage=='output':return encoder(images),ref['output_tokens']
            if stage=='batch':
                order=[1,0,1]
                return encoder(images[order]),ref['output_tokens'][order]
            raise ValueError(stage)
        for stage in (STAGES[1:] if args.stage=='all' else (args.stage,)):
            try:
                actual,expected=compute(stage); compare(stage,actual,expected);results.append(0)
            except NotImplementedError as exc:
                print(stage+' INCOMPLETE: '+str(exc));results.append(2)
            except Exception as exc:
                print(stage+' FAIL: '+str(exc));results.append(1)
    if 1 in results:return 1
    if 2 in results:
        print('Fill the TODOs in image_encoder.py, then check the next stage.')
        return 2
    print('Requested numerical checks passed. The reference file is never an encoder input.')
    return 0


if __name__=='__main__':
    sys.exit(main())
