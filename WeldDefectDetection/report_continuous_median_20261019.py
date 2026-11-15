"""Report a frozen median experiment, including unsuccessful ablations."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import run_continuous_median_20261019 as study
from raised_crown import learning as base
from circular_study import metrics

NAMES={'baseline':'Frozen 6 Oct median','profile0_sigma0':'Continuous plain',
       'profile0_sigma1':'Continuous + smoothing','profile1_sigma0':'Continuous + profile correction',
       'profile1_sigma1':'Continuous + correction + smoothing'}


def clipping_diagnostic(norm):
    records=[]
    for meta in study.read(study.OUT/'confirmation/index.json'):
        if meta['source_class'] not in ('open_keyhole','open_intrinsic'):continue
        with np.load(meta['npz_path']) as z:
            values=z['raw'][0,16:144][z['mask'][16:144]>0]
        records.append(dict(scene_id=meta['scene_id'],mask_pixels=len(values),
            high_clipped=int((values>=norm['high']).sum()),low_clipped=int((values<=norm['low']).sum())))
    result=dict(description='Post hoc input clipping diagnostic; no model or threshold changes',normalization=norm,
        total_pore_pixels=sum(r['mask_pixels'] for r in records),high_clipped_pore_pixels=sum(r['high_clipped'] for r in records),
        low_clipped_pore_pixels=sum(r['low_clipped'] for r in records),rows=records)
    study.write(study.OUT/'input_clipping_diagnostic.json',result)
    return result


def main():
    frozen=study.verify_frozen();result=study.read(study.OUT/'results.json')
    audit=study.read(study.OUT/'audit.json');assert audit['status']=='passed'
    selected=result['selected_arm'];rows=result['results'];arm=frozen['arms'][selected]
    diff=result['paired_differences_vs_baseline'][selected]
    for method in rows.values():
        assert len(method['rows'])==96
        assert method['primary_pores']['tp']+method['primary_pores']['fn']==48
        assert method['all_conditions']['tp']+method['all_conditions']['fn']==72
        assert method['primary_pores']['clean_rings']==24
    figures=ROOT/'docs/figures/continuous_median_20261019';figures.mkdir(parents=True,exist_ok=True)
    names=['baseline',*frozen['arms']]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,field,title in zip(axes,('primary_pores','all_conditions'),('Primary: clean + two open-pore classes','Secondary: including prescribed bulges')):
        vals=[rows[k][field]['f1'] for k in names]
        intervals=[rows[k]['primary_intervals' if field=='primary_pores' else 'all_intervals']['f1'] for k in names]
        colors=['#6b7280' if k=='baseline' else '#167b6b' if k==selected else '#537fa8' for k in names]
        ax.barh(np.arange(len(names)),vals,color=colors)
        # Percentile intervals need not enclose the point estimate in a small discrete sample.
        for i,(v,ci) in enumerate(zip(vals,intervals)):
            ax.plot(ci,[i,i],color='#18202a',lw=1.4)
            ax.text(min(v+.02,.96),i,f'{v:.3f}',va='center',fontsize=10)
        ax.set_yticks(np.arange(len(names)),[NAMES[k]+(' *' if k==selected else '') for k in names])
        ax.invert_yaxis();ax.set_xlim(0,1.07);ax.set_xlabel('Box F1, IoU ≥ 0.5');ax.set_title(title)
    fig.suptitle('Fresh confirmation: 24 independent backgrounds, 96 paired scenes\n* Selected on validation before confirmation; lines = 95% background bootstrap intervals',fontsize=12)
    fig.savefig(figures/'fresh_confirmation.png',dpi=170);plt.close(fig)

    norm=study.read(study.original.OUT/'normalization.json')
    clipping=clipping_diagnostic(norm)
    example_clipping=next(r for r in clipping['rows'] if r['scene_id']=='fresh-008-open_intrinsic')
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    detail={}
    # Fixed example index, chosen before looking at confirmation scores.
    pore_path=study.OUT/'confirmation/fresh-008-open_intrinsic.npz'
    with np.load(pore_path) as z:
        ys,xs=np.where(z['mask']);x=int(np.floor(xs.mean()/32)*32+16)
        y=int(np.clip(min(base.CENTRES,key=lambda c:abs(c-ys.mean())),48,112))
    for row,kind in enumerate(('clean','open_intrinsic')):
        path=study.OUT/f'confirmation/fresh-008-{kind}.npz'
        with np.load(path) as z:image=base.normalize(z['raw'][0],norm);truth=z['mask'][16:144]
        oldscore=base.median_image(image,frozen['baseline']['spec'])
        op=frozen['baseline']['selection']['best'];oldmask,oldboxes=metrics.binary(oldscore,op['threshold'],op['min_area'],op['closing'])
        residual=study.smooth(study.difference(image,arm['spacing'],arm['profile']),arm['sigma'])[16:144]
        score=abs(residual-np.array(arm['centre'],'f4')[:,None])/np.array(arm['scale'],'f4')[:,None]
        op=arm['best']['op'];mask,boxes=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
        tb=metrics.regions(truth)[1];oldm=metrics.match(oldboxes,tb,5024);newm=metrics.match(boxes,tb,5024)
        detail[kind]=dict(baseline=oldm,selected=newm)
        cut=lambda a:base.tile(a,y-16,x)
        bg=base.tile(image,y,x)
        for col,ax in enumerate(axes[row]):
            value=cut(truth) if col==1 else bg
            ax.imshow(value,origin='lower',cmap='gray',vmin=0,vmax=1,extent=(-1.6,1.6,-1.6,1.6))
            if col>=2:
                m=oldmask if col==2 else mask;crop=cut(m)
                if crop.any():ax.contour(np.linspace(-1.575,1.575,64),np.linspace(-1.575,1.575,64),crop,[.5],colors=['#f4a340'],linewidths=1.4)
                t=cut(truth)
                if t.any():ax.contour(np.linspace(-1.575,1.575,64),np.linspace(-1.575,1.575,64),t,[.5],colors=['#38dd9c'],linestyles='--',linewidths=1.2)
            title=['Normalized TOPO','Surface truth','Frozen median','Validation-selected continuous median'][col]
            if col>=2:
                m=oldm if col==2 else newm;title+=f'\nTP {m["tp"]} / FP {m["fp"]} / FN {m["fn"]}'
            ax.set_title(title,fontsize=10);ax.set_xlabel('Local arc offset (mm)')
            ax.set_ylabel(('Clean' if kind=='clean' else 'Intrinsic pore')+'\nLocal radial offset (mm)' if col==0 else 'Local radial offset (mm)')
    fig.suptitle('Predeclared example background fresh-008; counts refer to complete rings\nGreen dashed: truth; orange: prediction. Teaching crop uses truth only for framing.',fontsize=12)
    fig.savefig(figures/'fixed_example.png',dpi=170);plt.close(fig)
    study.write(study.OUT/'fixed_example.json',detail)

    primary=diff['primary']['f1'];delta=primary['difference'];lo,hi=primary['interval_95']
    verdict=('increased' if delta>0 else 'decreased' if delta<0 else 'was unchanged')
    evidence=('The paired interval excludes zero.' if lo>0 or hi<0 else 'The paired interval includes zero, so this experiment does not establish a reliable difference.')
    lines=['# Continuous circular-median experiment, 19 October 2026','',
        f'Completed campaign: `{study.OUT.name}`. Selected arm: **{NAMES[selected]}**, selected on validation before fresh confirmation.',
        '', f'Primary pore-detection F1 **{verdict}** from **{rows["baseline"]["primary_pores"]["f1"]:.3f}** to **{rows[selected]["primary_pores"]["f1"]:.3f}**. {evidence}',
        '', '## Question and controlled comparison','',
        'Does a one-pixel-step circular median of four preceding and four following samples improve the existing frozen neighbouring-patch median? The geometry, defect amplitudes, acquisition model and train-only input normalization are unchanged. The primary endpoint is strict box F1 at IoU >= 0.5 on clean plus the two open-pore conditions. Bulges are retained as a secondary assessment.',
        '', 'Development reuses only the original 24 clean training backgrounds for residual calibration and 24 validation backgrounds for selection. New confirmation contains 24 independent backgrounds, six in each dwell stratum, and 96 scenes (clean, keyhole-shaped opening, intrinsic opening and assumed bulge). Duplicate hidden controls are omitted. All methods see exactly the same fresh images.',
        '', '## Fresh confirmation results','',
        '| Method | Pore-task F1 | Precision | Recall | All-condition F1 | Clean rings flagged |',
        '|---|---:|---:|---:|---:|---:|']
    for k in names:
        p=rows[k]['primary_pores'];a=rows[k]['all_conditions']
        lines.append(f'| {NAMES[k]}'+(' **(selected)**' if k==selected else '')+f' | {p["f1"]:.3f} | {p["precision"]:.3f} | {p["recall"]:.3f} | {a["f1"]:.3f} | {p["clean_rings_flagged"]}/{p["clean_rings"]} |')
    for label,key in [('Primary pore-task','primary'),('All-condition','all_conditions')]:
        d=diff[key]['f1'];ci=d['interval_95']
        lines+=['',f'{label} selected-minus-baseline F1 difference: **{d["difference"]:+.3f}**, paired 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}]**.']
    lines+=['','![Fresh confirmation](figures/continuous_median_20261019/fresh_confirmation.png)',
        '', '## Selection and ablations','',
        'Four ablation arms vary continuous profile correction and residual Gaussian smoothing independently. Each arm chooses among spacings 16, 32 and 64 pixels using the same validation threshold, area and closing grid. A first-neighbour spacing of d produces offsets +/-d, +/-2d, +/-3d and +/-4d. Every angular pixel is evaluated, with circular wrapping. Profile correction uses a 65-column running median and a nine-row radial Gaussian, rather than blockwise correction.',
        '', '| Arm | Spacing (px / nominal mm) | Smoothing sigma | Threshold | Min area | Closing | Validation pore F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k,a in frozen['arms'].items():
        op=a['best']['op'];m=a['best']['summary']['primary_pores']
        lines.append(f'| {NAMES[k]} | {a["spacing"]} / {a["spacing"]*.05:.2f} | {a["sigma"]} | {op["threshold"]} | {op["min_area"]} | {op["closing"]} | {m["f1"]:.3f} |')
    lines+=['','All 12 spacing/arm candidates and all operating-point trials are retained under `development/`. The selected arm is not replaced by a different arm after observing confirmation.',
        '', '## Per-condition recovery','',
        '| Method | Keyhole opening recall | Intrinsic opening recall | Bulge recall | Pixel IoU, pore task |',
        '|---|---:|---:|---:|---:|']
    for k in names:
        b=rows[k]['by_class'];p=rows[k]['primary_pores']
        lines.append(f'| {NAMES[k]} | {b["open_keyhole"]["recall"]:.3f} | {b["open_intrinsic"]["recall"]:.3f} | {b["bulge"]["recall"]:.3f} | {p["pixel_iou"]:.3f} |')
    lines+=['','## Response versus footprint recovery','',
        'These are descriptive checks of frozen predictions, not additional threshold selection. Any overlap means at least one predicted pixel lies inside the geometric pore mask. It is weaker than recovering the pore footprint at the required box IoU.',
        '', '| Method | Correctly matched pores / 48 | Some mask overlap but no strict match | Neither match nor mask overlap |',
        '|---|---:|---:|---:|']
    for k in ('baseline',selected):
        pores=[r for r in rows[k]['rows'] if r['source_class'] in ('open_keyhole','open_intrinsic')]
        matched=sum(r['tp']>0 for r in pores)
        partial=sum(r['tp']==0 and r['intersection']>0 for r in pores)
        absent=sum(r['tp']==0 and r['intersection']==0 for r in pores)
        lines.append(f'| {NAMES[k]} | {matched}/48 | {partial} | {absent} |')
    lines+=['','## Acquisition strata','',
        'Each row contains six independent backgrounds and 18 primary-task scenes. These small-stratum results are descriptive.',
        '', '| Dwell time (microseconds) | Frozen median pore F1 | Selected continuous pore F1 |',
        '|---|---:|---:|']
    for b,(low,high) in enumerate(zip(study.original.EDGES[:-1],study.original.EDGES[1:])):
        values=[metrics.summary([r for r in study.subset(rows[k]['rows']) if r['dwell_bin']==b])['f1'] for k in ('baseline',selected)]
        lines.append(f'| {low:g} to {high:g} | {values[0]:.3f} | {values[1]:.3f} |')
    lines+=['','## Fixed example','',
        'Background fresh-008 was chosen for illustration before confirmation results were inspected. Counts refer to whole rings, not just the displayed crop.',
        '', '![Fixed example](figures/continuous_median_20261019/fixed_example.png)',
        '', f'In this example, both filters respond to the upper portion of the opening, but neither recovers a sufficiently large box. A post hoc check found {example_clipping["high_clipped"]} of its {example_clipping["mask_pixels"]} pore pixels clipped at the fixed input normalization ceiling. Across all 48 pore scenes, only {clipping["high_clipped_pore_pixels"]:,} of {clipping["total_pore_pixels"]:,} pore pixels ({100*clipping["high_clipped_pore_pixels"]/clipping["total_pore_pixels"]:.1f}%) are clipped high, so clipping in this illustration cannot explain the campaign-wide failure by itself. See `input_clipping_diagnostic.json`. No settings were changed in response to this check.',
        '', '## Decision','',
        'Do not replace the frozen baseline with this continuous variant on the strength of this experiment. None of the four continuous arms improved primary pore F1 on fresh confirmation. The selected arm produced a lower point estimate, with an interval spanning zero. Both methods remain weak at recovering complete pore footprints.',
        '', 'The next development experiment should separate anomaly localisation from footprint reconstruction, measuring whether a residual response finds the pore and then whether a region-construction step recovers its extent. Any new normalization, threshold or reconstruction rule must be selected on development data and assessed on another fresh confirmation set. This is a proposed next experiment, not a demonstrated fix.',
        '', '## Evidence and limits','',
        '- Numerical checks passed for circular shift equivariance, target exclusion from the plain eight-sample reference, a manual median calculation, cancellation of an angularly constant profile, and seam-spanning region connectivity. The corrected arm also uses a local profile containing the current angular position, and smoothing mixes nearby residuals.',
        '- The audit verifies source/data hashes, frozen selections, 96 scenes across 24 fresh backgrounds, no seed overlap with the earlier acquisition campaign, and unchanged historical artifacts.',
        '- Background-stratified paired bootstrap intervals use 2,000 replicates. They describe this finite synthetic experiment, not real-world uncertainty or industrial reliability.',
        '- Same-image differences are the valid improvement comparison. The old published 0.315 all-condition F1 is from a different confirmation population and is not subtracted from this experiment.',
        '- This experiment changes continuous reference construction, residual calibration and validation objective together relative to the historical baseline. It does not isolate scan stride alone. Within the new four-arm study, correction and smoothing are compared under the same search budget.',
        '- Mask labels describe geometric surface openings. Detector contrast may emphasize edges, and a local response can still fail strict footprint matching.',
        '- Declared ring coordinates are used. No localisation robustness, real-image transfer or physical calibration is established.',
        '', '## Reproduce','', '```powershell',
        '.venv\\Scripts\\python.exe WeldDefectDetection\\run_continuous_median_20261019.py all',
        '.venv\\Scripts\\python.exe WeldDefectDetection\\report_continuous_median_20261019.py', '```','',
        f'Artifacts: `{study.OUT}`. Main files: `protocol.json`, `self_checks.json`, `development/`, `frozen.json`, `sources_at_freeze.zip`, `confirmation/index.json`, `predictions/`, `results.json`, and `audit.json`.']
    target=ROOT/'docs/CONTINUOUS_MEDIAN_RESULTS_20261019.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    study.write(study.OUT/'report_artifacts.json',dict(report=str(target),sha256=study.sha(target),figures={str(p):study.sha(p) for p in figures.glob('*.png')}))
    print(target,flush=True)


if __name__=='__main__':main()
