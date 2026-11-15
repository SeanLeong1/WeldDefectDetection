"""Delivery checks beyond the frozen numerical evaluation."""
import platform
from pathlib import Path
import cv2,numpy,scipy,torch,matplotlib
from box_comparison.core import sha
from . import generation as g,evaluation


def run():
    frozen=evaluation.verify_freeze()
    assert frozen['cnn']['config']['channels']==1,'Headline comparison must be TOPO-only'
    assert len({c['variant'] for c in frozen['cnn_candidates']})==8
    assert len(frozen['median_candidates'])==6
    for variant in {'fixed_negatives',frozen['cnn_variant']}:
        assert {c['seed'] for c in frozen['cnn_candidates'] if c['variant']==variant}==set(g.SEEDS)
    freeze_time=(g.OUT/'frozen_models.json').stat().st_mtime
    observation=g.OUT/'freeze_observation.json'
    if observation.exists():
        assert sha(g.OUT/'frozen_models.json')==g.read(observation)['frozen_models_sha256']
    first_test=min(Path(r['npz_path']).stat().st_mtime for r in g.read(g.OUT/'datasets/test/index.json'))
    assert first_test>freeze_time,'Model file was changed after test generation began'
    scope_path=g.OUT/'cv_scope_amendment.json'
    if scope_path.exists():
        scope=g.read(scope_path);assert all(scope['reuse_checks'].values())
    assert g.read(g.OUT/'diagnostics/roi_equivalence.json')['status']=='passed'
    historical_path=g.OUT/'diagnostics/historical_transfer_integrity.json'
    if not historical_path.exists():
        previous=g.read(g.baseline.OUT/'frozen_models.json')
        assert sha(g.baseline.OUT/'normalization.json')==previous['normalization_sha256']
        g.write(historical_path,dict(normalization_sha256=sha(g.baseline.OUT/'normalization.json'),
            frozen_models_sha256=sha(g.baseline.OUT/'frozen_models.json'),checkpoints_verified=True))
    historical=g.read(historical_path)
    assert sha(g.baseline.OUT/'normalization.json')==historical['normalization_sha256']
    assert sha(g.baseline.OUT/'frozen_models.json')==historical['frozen_models_sha256']
    for spec in g.read(g.baseline.OUT/'frozen_models.json')['models'].values():
        if 'checkpoint' in spec:assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    for file in ('runtime_benchmark.json','diagnostics/tile_origin_sensitivity.json','diagnostics/error_taxonomy.json','diagnostics/pore_convergence.json'):
        assert (g.OUT/file).exists(),file
    result=dict(status='passed',primary_topo_only=True,cnn_variants=8,median_variants=6,
        replicated_variants=sorted({'fixed_negatives',frozen['cnn_variant']}),replicate_seeds=g.SEEDS,
        frozen_models_sha256=sha(g.OUT/'frozen_models.json'),
        model_file_mtime=freeze_time,first_test_array_mtime=first_test,
        chronology='Recorded model file predates every saved held-out scene; no post-test model edits.',
        scope_amendment_reuse_verified=True if scope_path.exists() else None,
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
                         torch=torch.__version__,opencv=cv2.__version__,matplotlib=matplotlib.__version__),
        scientific_limit='Synthetic evidence only. Small-pore response retains quantified sampling sensitivity. No measured SEM or acceptance validation.')
    g.write(g.OUT/'final_verification.json',result);print(result,flush=True)


if __name__=='__main__':run()
