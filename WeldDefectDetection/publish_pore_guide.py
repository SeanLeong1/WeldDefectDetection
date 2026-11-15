"""Publish the completed pore campaign into the current chronological teaching guide."""
from pathlib import Path
import hashlib,json,re
REPO=Path(__file__).resolve().parent.parent
ROOT=REPO/'DataSimulation/data/output/pore_campaign_20260921_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
    verification=json.loads((ROOT/'verification.json').read_text())
    if verification['status']!='ok':raise RuntimeError('Campaign verification must pass first.')
    guide=REPO/'DataSimulation/PROJECT_PIPELINE_CURRENT.md'
    previous=ROOT/'PROJECT_PIPELINE_BEFORE_PORE_CAMPAIGN.md'
    if not previous.exists():previous.write_bytes(guide.read_bytes())
    history=previous.read_text(encoding='utf-8').split('## Foundation and historical evidence',1)[1]
    lines=(ROOT/'REPORT.md').read_text(encoding='utf-8').splitlines()
    chapter=['## 29. Surface-expressed pore campaign','']
    for line in lines[2:]:
        if line.startswith('## '):line='### 29.'+line[3:]
        chapter.append(line)
    balanced=ROOT/'balanced_overfit_diagnostic.json'
    if balanced.exists():
        diagnostic=json.loads(balanced.read_text())
        chapter += ['', '### 29.8. Can the network fit the supplied masks?', '',
            'A supplementary diagnostic used four positive training targets of each pore type and eight clean training targets. It trained for 300 full-batch steps without augmentation. It was never a candidate for model selection.',
            f"Training-subset pixel F1 was {diagnostic['metrics']['keyhole_pore']['pixel_f1']:.3f} for keyhole pores and {diagnostic['metrics']['intrinsic_pore']['pixel_f1']:.3f} for intrinsic pores, with {diagnostic['metrics']['clean']['pixel_fp']} false-positive clean pixels.",
            'Successful memorization checks whether this architecture and optimizer can represent the supplied small training set. It does not show that the image contains enough stable information to generalize to unseen pores.']
    result=json.loads((ROOT/'results.json').read_text());primary=result['primary']
    ci=result['uncertainty']['paired_difference_intervals']['quiet_'+primary+'_minus_'+primary]['box_f1']
    chapter += ['', '### 29.9. What does removing noise change?', '',
        f"For the frozen primary method ({primary}), noisy confirmation box F1 was {result['metrics'][primary]['box_f1']:.3f}; its paired noiseless score was {result['metrics']['quiet_'+primary]['box_f1']:.3f}. The 95% paired sequence-bootstrap interval for noiseless minus noisy box F1 was [{ci[0]:.3f}, {ci[1]:.3f}].",
        'The same masks, geometry, normalization, checkpoint and thresholds were used in both versions. Removing noise also removes the configured acquisition perturbations, so this comparison cannot isolate individual noise sources.',
        ('The interval supports a positive detection benefit from removing these perturbations.' if ci[0]>0 else 'The interval does not establish a positive detection benefit from removing these perturbations.'),
        'Geometric-mask extent, image response and the learned decision rule can still limit performance. The pore-strength coefficients were not increased to obtain these scores.']
    candidates=json.loads((ROOT/'model_candidates.json').read_text())
    chapter += ['', '### 29.10. Training variation and confidence intervals', '',
        'The reported neural result uses the seed and checkpoint selected by validation, not an ensemble of three networks. The best validation score from each seed was:', '',
        '| Method | Seed | Selected epoch | Validation box F1 |', '|---|---:|---:|---:|']
    for method in ('cnn','cnn_median'):
        for fit in candidates[method]:
            chapter.append(f"| {method} | {fit['seed']} | {fit['epoch']} | {fit['metrics']['box_f1']:.3f} |")
    chapter += ['', 'Confirmation intervals resample whole sequences while keeping the trained model fixed. They quantify sampling uncertainty conditional on those frozen weights. They do not include all uncertainty from retraining, renderer assumptions or real-world domain shift.']
    prefix='''# From CASINO to TOPO defect boxes: current project guide

**Current CV scope: surface-expressed keyhole and intrinsic pores, plus clean welds.** Four methods produce bounding boxes on matched TOPO crops: thresholding, CNN, neighbouring-median residual and CNN + median.

The [pore campaign report](data/output/pore_campaign_20260921_v1/REPORT.md) contains the current independent results. Section 29 below explains its generation pilot, training, selection and confirmation in chronological order. Sections 1-28 retain the historical project explanation and dated results; their numerical tables describe different campaigns and populations.

Constant B remains assumed at 0.9206948532270666 for the 304SS, 60 keV synthetic domain. Surface-pore response is empirical and has not been validated against measured SEM. Neither successful software checks nor synthetic scores establish physical validity.

The original `PROJECT_PIPELINE_FROM_CASINO_TO_DEFECT_DETECTION.md` and previous campaign snapshots remain preserved.

## Foundation and historical evidence'''
    guide.write_text(prefix+history+'\n\n'+'\n'.join(chapter)+'\n',encoding='utf-8')
    snapshot=ROOT/'PROJECT_PIPELINE_CURRENT_SNAPSHOT.md';snapshot.write_bytes(guide.read_bytes())
    (ROOT/'current_guide_provenance.json').write_text(json.dumps(dict(current_path=str(guide),current_sha256=sha(guide),snapshot_sha256=sha(snapshot),previous_sha256=sha(previous),results_sha256=sha(ROOT/'results.json')),indent=2))
    print(guide)
