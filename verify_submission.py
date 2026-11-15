"""Check delivered hashes and replay two saved confirmation examples."""
from pathlib import Path
import json, sys
import submission

ROOT=Path(__file__).resolve().parent
def main():
    manifest=submission.read(ROOT/'SUBMISSION_MANIFEST.json')
    failures=[p for p,h in manifest['files'].items() if not (ROOT/p).is_file() or submission.sha(ROOT/p)!=h]
    if failures:raise AssertionError('Package hash failures: '+', '.join(failures))
    print(f"Verified {len(manifest['files'])} packaged file hashes.")
    if '--inference' not in sys.argv:return
    expected=submission.read(ROOT/'examples/expected_predictions.json')
    engine=submission.install_portable_inspector();inspector=engine.FinalInspector()
    for kind in ['open_intrinsic','clean']:
        result=inspector.infer(ROOT/f'examples/{kind}.npy')
        for saved,method in [('cnn','cnn'),('combined','median')]:
            row=next(r for r in expected[saved] if r['scene_id']==f'test-023-{kind}-c64-r0')
            reference=row.get('predicted_boxes',row.get('boxes'))
            if reference is None:raise ValueError('Saved prediction boxes unavailable')
            assert result['boxes'][method]==reference,(kind,method,result['boxes'][method],reference)
    print('Both methods exactly reproduced saved boxes on the two bundled examples.')

if __name__=='__main__':main()
