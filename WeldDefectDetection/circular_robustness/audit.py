"""Verify population identity, provenance and completeness without fitting."""
from pathlib import Path
import zipfile
import numpy as np
from circular_study import metrics
from . import campaign as c, native_inputs, assessment


def run():
    c.prepare();native_inputs.prepare()
    records=c.read(c.OUT/'index.json');receipts=c.read(c.OUT/'native_inputs/index.json')
    assert len(records)==len(receipts)==c.COUNT*len(c.VARIANTS)
    assert len({r['scene_id'] for r in records})==len(records)
    seeds={r['seed'] for r in records}
    historical={c.g.BASES[s]+i for s,n in c.g.COUNTS.items() for i in range(n)}
    assert not seeds & historical and len(seeds)==c.COUNT
    receipt_map={r['scene_id']:r for r in receipts};prediction_files=[];files={};groups={}
    pore_counts={'clean':0,'single_keyhole':1,'single_intrinsic':1,'pair_far':2,'pair_near':2,'pair_overlap':2,'triple':3}
    for meta in records:
        path=Path(meta['npz_path']);assert c.sha(path)==meta['npz_sha256']
        with np.load(path) as z:
            assert z['raw'].shape==(3,128,5024) and np.isfinite(z['raw']).all()
            instances=z['instances'];assert len(instances)==pore_counts[meta['variant']]
            assert np.array_equal(instances.any(axis=0),z['mask'])
            assert all(len(metrics.regions(m[32:96])[1])==1 for m in instances)
        receipt=receipt_map[meta['scene_id']]
        assert all(receipt['exact_reproduction'].values())
        assert receipt['original_scene_sha256']==meta['npz_sha256']
        native=Path(receipt['path']);assert c.sha(native)==receipt['sha256']
        pred=c.OUT/'predictions_native'/(meta['scene_id']+'.json')
        predictions=c.read(pred);count=20 if pore_counts[meta['variant']]<=1 else 2
        assert len(predictions)==count*4
        errors=c.PERTURBATIONS if count==20 else [c.PERTURBATIONS[0],c.PERTURBATIONS[-1]]
        assert set(predictions)=={f'{e["name"]}/{method}/{point}' for e in errors for method in ('cnn','median') for point in ('best','low_fa')}
        assert all(r['scene_id']==meta['scene_id'] and r['background_id']==meta['background_id'] for r in predictions.values())
        assert all(r['injected_instances']==pore_counts[meta['variant']] for r in predictions.values())
        for key,row in predictions.items():groups.setdefault(key,[]).append(row)
        prediction_files.append(pred)
        for p in (path,path.with_suffix('.json'),native,native.with_suffix('.json'),pred):
            files[p.relative_to(c.OUT).as_posix()]=c.sha(p)
    assert not list((c.OUT/'predictions').glob('*.json'))
    results=c.read(c.OUT/'results.json')['results'];assert set(results)==set(groups)
    for key,rows in groups.items():
        assert results[key]['overall']==assessment.summarize(rows)
        assert results[key]['single_pore_population']==assessment.summarize([r for r in rows if r['injected_instances']<=1])
    protocol=c.OUT/'protocol_native_v2.json'
    before=min(p.stat().st_mtime_ns for p in prediction_files)>protocol.stat().st_mtime_ns
    assert before,'Protocol chronology failed'
    # Retain both superseded and active extraction sources with their recorded hashes.
    sources=set()
    for name,key in [('protocol.json','source_hashes'),('protocol_native_v2.json','sources')]:
        for p,h in c.read(c.OUT/name)[key].items():
            assert c.sha(p)==h;sources.add(Path(p))
    sources.update(Path(__file__).parent.glob('*.py'))
    sources.add(Path(__file__).parents[1]/'run_circular_robustness_20261004.py')
    with zipfile.ZipFile(c.OUT/'analysis_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p.relative_to(c.ROOT).as_posix())
    evidence=dict(backgrounds=len(seeds),scenes=len(records),injected_pores=sum(pore_counts[r['variant']] for r in records),
        disjoint_from_v4_train_val_test=True,all_array_shapes_masks_instances_verified=True,
        all_native_rerenders_exact=True,active_protocol_precedes_predictions=before,
        superseded_extraction_predictions_used=False,frozen_v4_sources_and_models_verified=True,
        saved_aggregate_metrics_recomputed_exactly=True,
        analysis_source_sha256={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in sorted(sources)},
        prediction_scene_files=len(prediction_files),evaluation_views=sum(len(c.read(p))//4 for p in prediction_files),
        physical_validation=False)
    c.write(c.OUT/'audit.json',evidence);c.write(c.OUT/'dataset_manifest.json',dict(files=files))
    (c.OUT/'README.md').write_text(
        '# Circular robustness population, 4 October 2026\n\n'
        '168 scenes share 24 independent fresh backgrounds. Seven paired layouts retain the v4 circular profile. '
        'Unfamiliar profiles are excluded. The detector, normalization and both operating points remain frozen v4.\n\n'
        'Active extraction: protocol_native_v2.json. Original protocol.json and sources.zip preserve the superseded '
        'saved-annulus interpolation design. No predictions from that approximation are used. '
        'Native rerenders are verified exactly against datasets before direct Cartesian extraction.\n\n'
        'datasets: raw TOPO/Tx/Ty (3,128,5024), quiet TOPO, union masks and individual masks. '
        'Models consume only normalized TOPO. Central 64 rows define truth. Native TOPO files deliberately contain '
        'NaN outside their validated halo; the extractor rejects missing context.\n\n'
        'predictions_native: fixed-coordinate boxes and matching evidence. results.json: all conditions and both '
        'operating points. audit.json and dataset_manifest.json: verified counts and SHA-256 identities. '
        'investigation: post-hoc postprocessing and separate numerical resolution diagnostics. '
        'analysis_sources.zip: active and historical implementation.\n\n'
        'Primary matching uses connected union regions at IoU >=0.5. Individual-pore matching is secondary. '
        'Overlapping surrogate bodies are not a physical interaction simulation. No measured-image validity is claimed.\n',encoding='utf-8')
    print(evidence,flush=True)


if __name__=='__main__':run()
