"""Complete evidence report for the frozen second median experiment."""
from pathlib import Path
import json,html,time,platform,sys,re
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import median_scale_templates_20261115 as st
from report_median_exploration_20261115 import paired,failure_breakdown,simple_markdown,rect
from apply_median_scale_templates_20261115 import infer

LABELS=dict(baseline='Published corrected median',previous='Previous revised median',box_only='Box changes only',
    adaptive='Adaptive scale, distant reference',count_calibrated='Count-calibrated scale, distant reference',reference='Selected reference and scale',
    multiscale='Multiscale LoG',templates='Training-derived matched templates',selected='Shared validation-selected winner',per_count='Count-specific validation winners',
    control_proportional_padding='Frozen control: size-dependent padding',control_lower_floor='Frozen control: MAD floor 0.001')
PRIMARY=tuple(LABELS)

def fmt(v):return f'{v:.3f}'

def audit(results):
    frozen=st.verify(True);index=st.read(st.OUT/'datasets/test/index.json')
    assert len(index)==864 and len({r['seed'] for r in index})==48
    for r in index:
        assert st.sha(r['raw_path'])==r['raw_sha256'],r['scene_id']
    geometries={r['geometry_path']:r['geometry_sha256'] for r in index}
    for path,digest in geometries.items():assert st.sha(path)==digest,path
    for method,result in results.items():
        rr=result['rows'];assert [r['scene_id'] for r in rr]==[r['scene_id'] for r in index]
        for r in rr:
            match=st.ex.metrics.match(r['predicted_boxes'],r['truth_boxes'],5024)
            assert [match[k] for k in ('tp','fp','fn')]==[r[k] for k in ('tp','fp','fn')]
        assert st.ex.summary(rr)['macro_f1']==result['macro_f1']
    legacy_checks=[];oldcal=st.read(st.ex.OUT/'calibration.json')
    oldop=st.read(st.ex.OUT/'frozen.json')['policies']['classical']['op']
    for i in np.linspace(0,len(index)-1,12,dtype=int):
        r=index[i];maps=st.ex.score_maps(np.load(r['raw_path']),oldcal,('original_static',oldop['method']))
        for name,op in [('baseline',st.ex.ORIGINAL_OP),('previous',oldop)]:
            boxes=st.ex.reconstruct(st.ex.components(maps[op['method']],op['threshold'],op['closing']),op)
            assert boxes==results[name]['rows'][i]['predicted_boxes'],(name,r['scene_id'],'legacy replay')
            legacy_checks.append(dict(policy=name,scene_id=r['scene_id'],matched=True))
    # Replay the standalone, truth-free inference entry point for every family.
    replay=[];timing={}
    for name in PRIMARY:
        durations=[]
        for i in (0,7,14):
            r=index[i];raw=np.load(r['raw_path']);start=time.perf_counter();out=infer(raw,r['count_multiplier'],name);durations.append(time.perf_counter()-start)
            assert out['scored_boxes']==results[name]['rows'][i]['predicted_boxes'],(name,r['scene_id'])
            replay.append(dict(policy=name,scene_id=r['scene_id'],matched=True))
        timing[name]=dict(median_seconds=float(np.median(durations)),scenes=3,includes_dependency_verification=True)
    receipt=dict(passed=True,raw_hashes_checked=len(index),geometry_hashes_checked=len(geometries),methods_rescored=len(results),prediction_rows_rescored=sum(len(r['rows']) for r in results.values()),
        independent_backgrounds=48,freeze_utc=frozen['frozen_utc'],standalone_inference_replays=replay,legacy_implementation_replays=legacy_checks,timing=timing,
        runtime=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,opencv=st.cv2.__version__,torch=st.ex.base.torch.__version__),
        limits='No full regeneration replay or physical validation. Timing is a three-scene descriptive check, includes verification and is not a hardware benchmark.')
    receipt['script_hashes']={str(p):st.sha(p) for p in sorted((st.ROOT/'WeldDefectDetection').glob('*median_scale_templates_20261115.py'))}
    receipt['result_hashes']={str(st.OUT/n):st.sha(st.OUT/n) for n in ('confirmation.json','frozen.json','selection.json','premise_diagnostics.json')}
    st.write(st.OUT/'audit.json',receipt);return receipt

def amplitude(results):
    out={}
    for name in PRIMARY:
        out[name]={}
        for a in (0,.05,.1):
            rr=[r for r in results[name]['rows'] if r['count_multiplier']==64 and r['amplitude']==a]
            out[name][str(a)]=st.ex.summary(rr)['by_count']['64']
    return out

def region_errors(results):
    out={}
    for name,result in results.items():
        counts=dict(predictions=0,pores=0,any_hit=0,unmatched_overlap=0,unmatched_no_overlap=0)
        for r in result['rows']:
            if r['count_multiplier']!=64:continue
            truth=r['truth_boxes'];pred=r['predicted_boxes']
            values=[st.ex.metrics.circular_iou(b,truth[0],5024) for b in pred] if truth else [0.]*len(pred)
            best=max(values,default=0.);winner=int(np.argmax(values)) if values else -1
            counts['predictions']+=len(pred);counts['pores']+=len(truth);counts['any_hit']+=int(best>0)
            for i,v in enumerate(values):
                if best>=.5 and i==winner:continue
                counts['unmatched_overlap' if v>0 else 'unmatched_no_overlap']+=1
        counts['any_overlap_f1']=2*counts['any_hit']/max(counts['predictions']+counts['pores'],1)
        out[name]=counts
    return out

def figures(results,breakdown):
    names=['baseline','previous','selected','control_proportional_padding','control_lower_floor','multiscale','templates']
    fig,ax=plt.subplots(figsize=(11,5.5));x=np.arange(3);width=.105
    for j,k in enumerate(names):ax.bar(x+(j-3)*width,[results[k]['by_count'][str(c)]['f1'] for c in (1,4,64)],width,label=LABELS[k])
    ax.set_xticks(x,['1×','4×','64×']);ax.set_ylim(0,1);ax.set_ylabel('Box F1, IoU ≥ 0.5');ax.set_xlabel('Effective-count multiplier')
    ax.set_title('Fresh confirmation: frozen primary policy and predefined controls');ax.legend(fontsize=8,loc='upper left',ncol=2);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(st.REPORT/'confirmation_f1.png',dpi=180);plt.close(fig)
    names=['baseline','previous','box_only','reference','multiscale','templates']
    fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(len(names))
    for field,label,col in [('correct','IoU ≥ 0.5','#2b8cbe'),('overlap_below_iou_threshold','Overlap, IoU < 0.5','#fdbb84'),('no_overlap','No overlap','#bdbdbd')]:
        v=np.array([breakdown[k]['64'][field] for k in names]);ax.bar(np.arange(len(names)),v,bottom=bottom,label=label,color=col);bottom+=v
    ax.set_xticks(np.arange(len(names)),['Published','Previous','Boxes only','Reference','Multiscale','Templates']);ax.set_ylim(0,205);ax.set_ylabel('Pore acquisitions at 64×');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.13),ncol=3)
    fig.tight_layout();fig.savefig(st.REPORT/'localisation_64x.png',dpi=180);plt.close(fig)
    metas={r['scene_id']:r for r in st.read(st.OUT/'datasets/test/index.json')}
    original={r['scene_id']:r for r in results['baseline']['rows']};selected={r['scene_id']:r for r in results['selected']['rows']};previous={r['scene_id']:r for r in results['previous']['rows']}
    pool=[r for r in selected.values() if r['count_multiplier']==64 and r['truth_boxes'] and r['repeat']==0]
    chosen=[]
    for label,rows,reverse in [('Largest improvement',[r for r in pool if r['tp'] and not original[r['scene_id']]['tp']],True),
        ('Smallest improvement',[r for r in pool if r['tp'] and not original[r['scene_id']]['tp']],False),
        ('Largest remaining miss',[r for r in pool if not r['tp']],True),
        ('Largest regression',[r for r in pool if not r['tp'] and original[r['scene_id']]['tp']],True)]:
        if rows:
            r=sorted(rows,key=lambda r:(r['diameter_mm'],r['scene_id']),reverse=reverse)[0]
            if r['scene_id'] not in [x[1]['scene_id'] for x in chosen]:chosen.append((label,r))
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,3.4*len(chosen)),squeeze=False);case_records=[]
    for aa,(label,r) in zip(axes,chosen):
        meta=metas[r['scene_id']];truth=r['truth_boxes'][0];cx=(truth[0]+truth[2])/2;cy=(truth[1]+truth[3])/2
        cols=(np.arange(-30,31)+int(round(cx)))%5024;top=max(0,int(cy)-25);bottom=min(128,top+50)
        image=st.ex.base.normalize(np.load(meta['raw_path']),st.ex.NORM)[16:144]
        for ax,k,title in zip(aa,('baseline','previous','selected'),('Published median','Previous revision','New shared selected method')):
            crop=image[top:bottom,cols];ax.imshow(crop,origin='lower',extent=(-30.5,30.5,top,bottom),cmap='gray',vmin=float(crop.min()),vmax=float(crop.max()))
            rect(ax,truth,cx,'lime');pred={'baseline':original,'previous':previous,'selected':selected}[k][r['scene_id']]
            for b in pred['predicted_boxes']:rect(ax,b,cx,'cyan')
            ax.set_xlim(-30.5,30.5);ax.set_ylim(top,bottom);ax.set_title(title,fontsize=10);ax.set_xlabel('Angular offset, pixels')
        aa[0].set_ylabel(f'{label}\nD = {r["diameter_mm"]:.3f} mm\nScored radial row')
        case_records.append(dict(reason=label,scene_id=r['scene_id'],diameter_mm=r['diameter_mm']))
    fig.suptitle('Illustrations selected after evaluation; green = truth, cyan = predictions',fontsize=11)
    fig.tight_layout();fig.savefig(st.REPORT/'examples_64x.png',dpi=160);plt.close(fig)
    pad={r['scene_id']:r for r in results['control_proportional_padding']['rows']}
    floor={r['scene_id']:r for r in results['control_lower_floor']['rows']}
    choices=[]
    for label,rr in [('Padding improves localisation',[r for r in pool if not r['tp'] and pad[r['scene_id']]['tp']]),
        ('Padding loses localisation',[r for r in pool if r['tp'] and not pad[r['scene_id']]['tp']])]:
        if rr:choices.append((label,max(rr,key=lambda r:(r['diameter_mm'],r['scene_id']))))
    fig,axes=plt.subplots(len(choices),3,figsize=(12,3.7*len(choices)),squeeze=False)
    for aa,(label,r) in zip(axes,choices):
        truth=r['truth_boxes'][0];cx=(truth[0]+truth[2])/2;cy=(truth[1]+truth[3])/2
        cols=(np.arange(-30,31)+int(round(cx)))%5024;top=max(0,int(cy)-25);bottom=min(128,top+50)
        img=st.ex.base.normalize(np.load(metas[r['scene_id']]['raw_path']),st.ex.NORM)[16:144];crop=img[top:bottom,cols]
        for ax,lookup,title in zip(aa,(selected,pad,floor),('Validation-selected policy','Frozen padding control','Frozen lower-floor control')):
            ax.imshow(crop,origin='lower',extent=(-30.5,30.5,top,bottom),cmap='gray',vmin=float(crop.min()),vmax=float(crop.max()))
            rect(ax,truth,cx,'lime')
            for b in lookup[r['scene_id']]['predicted_boxes']:rect(ax,b,cx,'cyan')
            ax.set_xlim(-30.5,30.5);ax.set_ylim(top,bottom);ax.set_title(title,fontsize=10);ax.set_xlabel('Angular offset, pixels')
        aa[0].set_ylabel(f'{label}\nD = {r["diameter_mm"]:.3f} mm\nScored radial row')
        case_records.append(dict(reason=label,scene_id=r['scene_id'],diameter_mm=r['diameter_mm'],figure='controls_64x.png'))
    fig.suptitle('Predefined controls; illustrated cases chosen after evaluation. Green = truth, cyan = prediction',fontsize=10)
    fig.tight_layout();fig.savefig(st.REPORT/'controls_64x.png',dpi=180);plt.close(fig)
    return case_records

def validation_sizes():
    target=st.OUT/'validation_size_analysis.json'
    if target.exists():return st.read(target)
    caches=[st.read(st.OUT/'validation_cache'/f"{r['scene_id']}.json") for r in st.read(st.ex.OLD/'datasets/val/index.json')]
    policies=st.read(st.OUT/'frozen.json')['policies']
    rr={k:dict(rows=st.evaluate(caches,policies[k]['op'])) for k in ('baseline','previous','selected','control_proportional_padding','control_lower_floor')}
    result=failure_breakdown(rr);st.write(target,result);return result

def main():
    st.REPORT.mkdir(exist_ok=True);results=st.read(st.OUT/'confirmation.json');selection=st.read(st.OUT/'selection.json')
    receipt=audit(results);breakdown=failure_breakdown(results);amps=amplitude(results);region_counts=region_errors(results);val_sizes=validation_sizes()
    differences={name:{base:paired(results[name]['rows'],results[base]['rows'],20261116) for base in ('baseline','previous','selected')} for name in results if name not in ('baseline','previous')}
    cases=figures(results,breakdown);diag=st.read(st.OUT/'premise_diagnostics.json');policies=selection['policies']
    d64=differences['selected']['baseline']['64'];old64=differences['selected']['previous']['64']
    out=dict(results={k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in results.items()},paired_differences=differences,
             failure_breakdown=breakdown,validation_size_breakdown=val_sizes,by_amplitude=amps,region_errors_64x=region_counts,selected_cases=cases,protocol=st.read(st.OUT/'protocol.json'),audit=receipt)
    st.write(st.REPORT/'results.json',out)
    lines=['# Separate median experiment: boxes, scales, templates and references','',
      '15 November 2026. All numerical comparisons below use the same fresh confirmation scenes unless explicitly labelled validation or historical. The renderer and previous experiments were preserved.','',
      '## Principal result','',
      f"The shared validation-selected method is **{policies['selected']['op']['family']}**, with operating point `{json.dumps(policies['selected']['op'],sort_keys=True)}`.",
      f"At 64× it achieves F1 **{fmt(results['selected']['by_count']['64']['f1'])}**, compared with **{fmt(results['baseline']['by_count']['64']['f1'])}** for the published corrected median and **{fmt(results['previous']['by_count']['64']['f1'])}** for the previous revision.",
      f"The paired gain over the published median is **{d64['delta']:+.3f}**, 95% background-bootstrap interval **[{d64['ci95'][0]:+.3f}, {d64['ci95'][1]:+.3f}]**. Relative to the previous revision it is **{old64['delta']:+.3f} [{old64['ci95'][0]:+.3f}, {old64['ci95'][1]:+.3f}]**.",
      'A predefined padding control, frozen before confirmation, performs better on this fresh sample: **64× F1 0.607**, with **108 TP and 56 FP**, versus **84 TP and 95 FP** for the previous revision. Its paired difference from that revision is **+0.154 [0.046, 0.253]**. This is an exploratory control comparison with an unadjusted interval, not the validation-selected primary result. No setting was changed after seeing confirmation.',
      'A positive point difference is not automatically a confirmed improvement: inspect the interval and the per-count precision/recall results below. No new CNN was trained or evaluated in this median-only experiment.','',
      '## What was tested','',
      '- Box formation: component gaps 0, 4 and 10 pixels; minimum areas 3 and 8; original or row-dependent shifts; fixed or size-proportional padding; union extent or width-derived height.',
      '- Score calibration: per-image row MAD and count-specific clean-training row MAD; floors 0.003, 0.001 and 0.0003; thresholds 4, 8, 12, 16 and 24.',
      '- Reference construction: original offsets ±64/128/192/256; previous ±32/48/64/80; closer guarded ±16/24/32/48; robust periodic fit with harmonics 1 through 5.',
      '- Multiscale response: sigma-squared-normalised Laplacian at sigma 1, 2, 4 and 8; maxima over space and scale; candidate box size from the winning scale, with validation-selected extent factor and aspect.',
      '- Matched templates: 18 signed, unit-energy average training responses from paired quiet pore/clean images, grouped by pore family, diameter group and radial position. Prototype widths and heights use training masks. These are label-informed classical templates, not an unsupervised median filter.',
      '- Count-specific selection: select a separate policy for each known count level, alongside the single shared policy.',
      '',f"Development used the original 96 training and 32 validation backgrounds. {selection['candidate_count']} configuration evaluations, including comparators and locked controls, are recorded, representing {len({json.dumps(r['op'],sort_keys=True) for r in selection['trials']})} distinct serialized operating points. Duplicate operating points can occur across stages. The search began sequentially: select boxes, then tune scores/references with those boxes fixed. A documented development-only amendment then revisited all 72 box settings at the best initial score operating point in each reference/calibration branch. This avoided forcing the original-score box optimum onto a different reference. It is not the Cartesian product of every possible setting.",
      'All policies and dependencies were frozen before generating 48 previously unused backgrounds, yielding 864 scenes: 192 pore and 96 clean acquisitions per count. Validation was reused from earlier work, so this remains exploratory rather than a preregistered replication.','',
      'The scale-space implementation follows the normalised-derivative/extremum principle described by [Lindeberg](https://www.csc.kth.se/cvap/abstracts/cvap198.html). Its box-size conversion is an experimental approximation for pore contrast, not a theorem about opening boundaries. The sigma-1 input preprocessing remains in place before the additional LoG scales.','',
      '## Fresh confirmation results','',
      '| Method | 1× F1 | 4× F1 | 64× F1 | Mean F1 | 64× clean alarms |',
      '|---|---:|---:|---:|---:|---:|']
    for name in PRIMARY:
        r=results[name];v=r['by_count'];lines.append(f"| {LABELS[name]} | {fmt(v['1']['f1'])} | {fmt(v['4']['f1'])} | {fmt(v['64']['f1'])} | {fmt(r['macro_f1'])} | {v['64']['clean_alarms']}/{v['64']['clean_scenes']} |")
    lines+=['','![Fresh confirmation results](confirmation_f1.png)','',
      'The shared winner was chosen using mean validation F1, not whichever method happens to score best on confirmation. In this run it is the selected reference/component policy, so those two rows coincide. The count-specific winners also coincide with that shared policy. These are policy-selection roles, not independent methods. Family-level policies are all shown to avoid selecting only favourable results.','',
      'The padding control and lower-floor control are shown because they test the requested mechanisms. Their stronger confirmation scores do not retroactively make them the validation-selected winners.','',
      '### Recall and precision across count levels','',
      '| Count | Method | TP | FP | FN | Precision | Recall |','|---|---|---:|---:|---:|---:|---:|']
    for c in (1,4,64):
        for name in ('baseline','previous','selected','control_proportional_padding','control_lower_floor'):
            v=results[name]['by_count'][str(c)];lines.append(f"| {c}× | {LABELS[name]} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | {v['recall']:.3f} |")
    lines+=['','### Paired differences from published median','',
      '| Method | 1× ΔF1 [95% interval] | 4× ΔF1 [95% interval] | 64× ΔF1 [95% interval] |','|---|---|---|---|']
    for name in PRIMARY[2:]:
        cells=[]
        for c in (1,4,64):
            d=differences[name]['baseline'][str(c)];cells.append(f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        lines.append('| '+LABELS[name]+' | '+' | '.join(cells)+' |')
    lines+=['','Intervals use 2,000 paired background resamples, retaining all related scenes and repeats together. They condition on the frozen selections, omit simulator and selection uncertainty, and are not adjusted for the multiple exploratory comparisons.','',
      '### Counts and localisation at 64×','',
      '| Method | TP | FP | FN | Precision | Recall | Missed with overlap | Missed without overlap |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in PRIMARY:
        v=results[name]['by_count']['64'];b=breakdown[name]['64'];lines.append(f"| {LABELS[name]} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | {v['recall']:.3f} | {b['overlap_below_iou_threshold']} | {b['no_overlap']} |")
    lines+=['','![Localisation outcome breakdown](localisation_64x.png)','',
      '| Method | Pores with any overlap | Strict FP that overlap a pore | Strict FP with no overlap | Any-overlap F1 |',
      '|---|---:|---:|---:|---:|']
    for name in ('baseline','previous','reference','control_proportional_padding','control_lower_floor','multiscale','templates'):
        r=region_counts[name];lines.append(f"| {LABELS[name]} | {r['any_hit']}/{r['pores']} | {r['unmatched_overlap']} | {r['unmatched_no_overlap']} | {r['any_overlap_f1']:.3f} |")
    lines+=['','Overlapping false-positive regions include insufficient-IoU boxes and extra boxes competing for one truth object. These counts distinguish localisation/duplication from responses away from the labelled opening; they do not establish a unique algorithmic cause.','',
      '### Pore-size recall at 64×','',
      '| Method | Below 0.3 mm | 0.3–0.6 mm | At least 0.6 mm |','|---|---:|---:|---:|']
    for name in PRIMARY:
        b=breakdown[name]['64']['by_diameter'];lines.append('| '+LABELS[name]+' | '+' | '.join(f"{v['tp']}/{v['total']}" for v in b.values())+' |')
    lines+=['','These are acquisition-level counts, including repeated views of the same geometry, not independent pore specimens.','',
      '### Validation-to-confirmation transfer','',
      '| Method | Validation mean F1 | Confirmation mean F1 | Validation 64× F1 | Confirmation 64× F1 |','|---|---:|---:|---:|---:|']
    for name in PRIMARY:
        if name=='per_count':continue
        a=policies[name];b=results[name];lines.append(f"| {LABELS[name]} | {a['macro_f1']:.3f} | {b['macro_f1']:.3f} | {a['by_count']['64']['f1']:.3f} | {b['by_count']['64']['f1']:.3f} |")
    lines+=['','Differences can reflect selection optimism and sampling variation. This experiment does not isolate their respective contributions.','',
      '### Why the validation ranking needs caution','',
      'Validation contains 60 small, 44 medium and 24 large pore acquisitions per count. Fresh confirmation contains 60, 52 and 80 respectively. Large pores therefore account for 18.8% of validation pore acquisitions but 41.7% of fresh pore acquisitions. The large-pore validation group spans only six independent backgrounds, compared with twenty in fresh confirmation.',
      'Radial-position composition also differs: the large-pore validation group has 4 inner-flank, 16 centre and 4 outer-flank acquisitions, versus 28/28/24 in fresh confirmation. These are observational differences between samples, not controlled causal interventions.',
      '| Policy | Validation small | Validation medium | Validation large | Fresh small | Fresh medium | Fresh large |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('previous','selected','control_proportional_padding','control_lower_floor'):
        counts=[f"{v['tp']}/{v['total']}" for source in (val_sizes,breakdown) for v in source[name]['64']['by_diameter'].values()]
        lines.append('| '+LABELS[name]+' | '+' | '.join(counts)+' |')
    lines+=['','These are recall counts, not size-specific F1. The reversal is not explained by size frequencies alone: performance within size groups also changes. A future development set should cover size and radial-position combinations across more independent backgrounds, and report size-stratified performance alongside the aggregate objective. Keep this confirmation set out of that new selection process.','',
      '### Reference alternatives on validation','',
      '| Reference/calibration branch | Best mean validation F1 | Best operating point |','|---|---:|---|']
    for prefix in ('long_adaptive','long_count','short_adaptive','near_adaptive','periodic_adaptive'):
        rows=[r for r in selection['trials'] if r['op']['family']=='components' and r['op']['score'].startswith(prefix)]
        best=st.choose(rows);lines.append(f"| {prefix} | {best['macro_f1']:.3f} | `{json.dumps(best['op'],sort_keys=True)}` |")
    lines+=['','These are development results after a bounded search, not fresh confirmation for every reference branch. Only frozen policies listed in the confirmation table were evaluated as confirmation comparisons.','',
      '## Checking the proposed explanation','',
      '### Historical 64× boxes','',
      'Direct inspection confirms 147 of 192 pore acquisitions have an overlapping prediction. There are 161 predicted regions, all of which touch a pore. Strict IoU ≥ 0.5 matching yields 68 TP, 93 FP and 124 FN. Relaxing matching to any overlap yields 147 TP and 14 FP; those 14 are excess/duplicate predictions, not off-pore regions. Thus all 93 strict false-positive regions have some overlap, rather than only approximately 79.',
      'The 79 newly matched acquisitions are the increase in matched pores under the relaxed rule. They are not a complete classification of all false-positive boxes. The resulting F1 is 0.832861. This is a conditional value with the existing prediction count, not a hard ceiling after merging duplicates, dropping regions or recovering previously missed pores.',
      'A box that touches a pore is evidence of a response near it, but can still be a poor opening estimate. One unmatched box plus one missed pore contributes both FP and FN under the stated detection metric. Relaxing IoU changes the task being measured; it does not repair localisation. Fixed pipeline settings also do not make their outputs independent of noise: noise can still change threshold support, connectivity and box geometry.','',
      '### Score floor and structure','',
      'The 0.003 floor implies a minimum residual cutoff of 0.048 at threshold 16. The actual original per-row cutoff has minimum/median/maximum approximately **0.048/0.0683/0.0909**. Therefore the score is not a uniform 0.048 test.',
      'The raw-image repeat-variability ratio does not directly determine how a normalised residual threshold should scale. Square-root normalisation, preprocessing and persistent surface residuals matter. Per-image MAD is robust to sparse pore pixels, but it estimates residual spread including structure, and a small pore fraction alone does not guarantee suitable calibration.','',
      '| Reference | 64× clean residual RMS | Quiet clean residual RMS | Median paired quiet-residual/repeat-noise RMS | Median fraction of rows at 0.003 floor |','|---|---:|---:|---:|---:|']
    noise=diag['noise_structure'];v=noise['by_count']['64']
    for ref in st.REFS:
        d=v[ref];lines.append(f"| {ref} | {d['residual_RMS']:.6f} | {d['quiet_residual_RMS']:.6f} | {d['quiet_to_repeat_RMS']:.3f} | {d['floor_active_row_fraction']:.3f} |")
    lines+=['',f"The median repeat-noise RMS is {v['repeat_noise_RMS']:.6f} in the same preprocessed normalised coordinates. Ratios are medians of paired ratios, not ratios of displayed medians. These diagnostics use 32 clean validation backgrounds and remove no regions using test labels.",
      'The distant reference leaves substantial quiet-image structure, and closer references reduce it in this diagnostic. This supports testing reference bias. It does not establish that crown harmonic amplitude alone causes the detector failures.','',
      '### Crown-variation amplitude','',
      '| Amplitude a | Historical 64× median F1 | Historical pore acquisitions | Fresh baseline F1 | Fresh shared-selected F1 |','|---|---:|---:|---:|---:|']
    for a in ('0','0.05','0.1'):
        h=diag['historical']['by_amplitude'][a];lines.append(f"| {a} | {h['f1']:.3f} | {h['pore_acquisitions']} | {amps['baseline'][a]['f1']:.3f} | {amps['selected'][a]['f1']:.3f} |")
    lines+=['','These groups contain different backgrounds and pore sizes, so they are descriptive associations rather than a controlled amplitude ablation. The historical scores increase with amplitude, contrary to a simple claim that larger amplitude necessarily explains poorer median performance.','',
      '## Locked component ablations','',
      'Each displayed control changes one setting of the selected reference/component policy without retuning. Locked controls that reproduce its existing setting are omitted from this table as exact duplicates, but remain in the evidence files. This provides a more direct comparison than subtracting results from separately optimised families. It is not an ablation of the template method.','',
      '| Policy | 1× F1 | 4× F1 | 64× F1 | 64× difference from selected [95% interval] |','|---|---:|---:|---:|---|']
    for name in ['reference']+[k for k in results if k.startswith(('ablation_','control_')) and policies[k]['op']!=policies['reference']['op']]:
        d=differences[name]['selected']['64'];lines.append('| '+name+' | '+' | '.join(fmt(results[name]['by_count'][str(c)]['f1']) for c in (1,4,64))+f" | {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    lines+=['',
      'The measured quiet-contrast centroid displacement is not necessarily the displacement of a thresholded component box. Consequently, substituting those centroid offsets for a box correction requires an empirical test. Likewise, a scale-normalised extremum sizes a contrast feature, which need not equal the whole opening. These distinctions explain why the proposals are plausible experiments rather than guaranteed improvements; the tables establish performance, not a unique causal explanation for each failure.',
      '', '## Frozen operating points','']
    for name in PRIMARY:
        p=policies[name];ops={c:v['op'] for c,v in p.items()} if name=='per_count' else p['op']
        lines.append(f'- **{LABELS[name]}:** `{json.dumps(ops,sort_keys=True)}`')
    lines+=['',
      'Merging uses Euclidean distance between component bounding rectangles, with a maximum union width/height of 40 pixels to limit chains of unrelated regions. Fragments smaller than 3 pixels are removed by the existing component extractor; the selected area requirement is applied after merging. The row-dependent correction interpolates +1.54 pixels at the inner flank, zero at the centre and −1.17 at the outer flank. It uses the previously measured training diagnostic and does not snap boxes to three discrete rows. The original fixed-inward rule already gives zero shift exactly at row 64; the proposed interpolation changes its behaviour around the centre and on the flanks.',
      'Proportional padding is 1 pixel plus the selected fraction of the larger component dimension per side. Width-derived height imposes a fixed aspect prior. Multiscale/template candidates undergo periodic nonmaximum suppression at IoU 0.2, with no top-k truncation. Templates are evaluated across the whole ring without supplying a pore location. Their labelled training geometry remains a substantive prior.','',
      '## Illustrative cases','',
      '![Selected examples](examples_64x.png)','',
      'Examples were selected after evaluation using fixed descriptive rules: largest/smallest improvement, largest remaining miss and largest regression where available. They illustrate mechanisms and do not estimate frequencies. Exact identities are in results.json.','',
      '![Padding improvements and regressions](controls_64x.png)','',
      'The additional control figure shows the largest padding improvement and largest padding regression, where available. This illustrates the extent trade-off without hiding a less favourable case.','',
      '## Decision supported by this experiment','',
      ('The selected method improves 64× F1 over the published corrected median with a paired interval entirely above zero.' if d64['ci95'][0]>0 else 'The selected method does not establish a positive 64× improvement over the published median at the reported paired interval.'),
      ('It also establishes an incremental 64× improvement over the previous revision within this experiment.' if old64['ci95'][0]>0 else 'The incremental 64× improvement over the previous revision is not established: its paired interval is not entirely above zero. Do not describe the validation advantage as a confirmed upgrade.'),
      'The frozen padding control is the strongest next candidate for confirmation. At 64× it improves large-pore correct localisations from 17/80 under the previous revision to 42/80, while small-pore correct localisations fall from 26/60 to 22/60. Lowering the MAD floor reaches 66/80 large pores but only 16/60 small pores, illustrating a different trade-off.',
      'The padding and lower-floor controls both ranked worse on validation but better on fresh confirmation than the primary selected policy. This ranking reversal is evidence of unstable transfer across these finite samples. More large pores occur in this fresh set than in the preceding confirmation set, which can affect aggregate scores; it does not by itself prove the cause of the reversal. Do not combine the better-performing controls or retune them on this confirmation set and call the outcome independently validated.',
      'For a thesis, report the primary policy and the favourable frozen controls separately. To promote the padding control to the default method, keep its current settings fixed and evaluate it on another untouched sample. The present results support further testing and establish observed synthetic gains; they do not establish which policy will consistently perform best.',
      'Choose an operating point with the required recall/false-alarm trade-off in mind. An F1 gain accompanied by lower low-count recall is not an improvement in every practical detection objective. Keep the published historical comparison intact and report this experiment separately.',
      ('The count-specific selector chose the same operating point at all three levels in this bounded candidate set. It therefore adds no benefit over the shared selected policy here.' if len({json.dumps(v['op'],sort_keys=True) for v in policies['per_count'].values()})==1 else 'The count-specific selector chose different policies; their separate results should be reported without assuming that specialisation improves transfer.'),
      '', '## Reproduction and limits','',
      '- Main script: `WeldDefectDetection/median_scale_templates_20261115.py`. Run `prepare` and `development`, then the refinement and lock scripts below, then `generate` and `confirm`, using the project virtual environment. Do not refreeze after confirmation begins.',
      '- Development amendment: `WeldDefectDetection/refine_median_scale_templates_20261115.py`. Final controls and freeze: `WeldDefectDetection/lock_median_scale_templates_20261115.py`. The protocol preserves the timing and reason for the amendment.',
      '- Optional concurrent evaluation: `WeldDefectDetection/stream_median_scale_templates_20261115.py` evaluates completed backgrounds using frozen policies while generation continues. The final `confirm` action assembles all results.',
      '- Diagnostics: `WeldDefectDetection/diagnose_median_scale_templates_20261115.py`.',
      '- Report and verification: `WeldDefectDetection/report_median_scale_templates_20261115.py`.',
      '- Truth-free inference: `WeldDefectDetection/apply_median_scale_templates_20261115.py input.npy --count 64 --policy selected`. The array must be raw registered TOPO with shape 160 × 5024. Output includes scored and raw-coordinate cyclic boxes.',
      '- Full evidence: `DataSimulation/data/output/median_scale_templates_20261115_v1`, including protocol, templates, training calibration, all validation trials, frozen policies, per-scene confirmation predictions, diagnostics and audit.',
      f"- Audit passed: {receipt['raw_hashes_checked']} image hashes, {receipt['prediction_rows_rescored']} prediction records independently re-scored, and {len(receipt['standalone_inference_replays'])} standalone inference replays.",
      '- All performance remains synthetic and assumes nominal ring coordinates. The experiment preserves the same renderer, restricted pore families and one-pore scene design. It establishes algorithmic performance under that distribution, not physical correctness or transfer to measured welds.',
      '- A template improvement cannot be described as a gain from median filtering alone. Template dimensions and responses are learned from labelled synthetic training geometries. The distinction is a learned template model versus a CNN, not labelled versus unlabelled learning.',
      '- The finite, sequential search can miss interactions and stronger settings. A negative result for this multiscale or periodic implementation does not disprove those method classes.',
      '- Timing checks are descriptive only, use three scenes per policy and include dependency verification. Detailed values are in audit.json.']
    text='\n'.join(lines)+'\n';(st.REPORT/'REPORT.md').write_text(text,encoding='utf-8')
    body=re.sub(r'\[([^\]\n]+)\]\((https?://[^)\s]+)\)',r'<a href="\2">\1</a>',simple_markdown(text))
    document='<!doctype html><html><head><meta charset="utf-8"><title>Separate median experiment</title><style>body{font:16px/1.55 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#19232b}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #ccd6de;padding:7px;text-align:left}th{background:#eaf1f6}img{max-width:100%}code{overflow-wrap:anywhere}h1,h2{color:#174d71}li{margin-bottom:8px}</style></head><body>'+body+'</body></html>'
    (st.REPORT/'index.html').write_text(document,encoding='utf-8')
    print(json.dumps({k:v for k,v in out['results'].items() if k in PRIMARY},indent=2),flush=True)

if __name__=='__main__':main()
