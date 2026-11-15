"""Summarise frozen follow-up; report all policies and conditional uncertainty."""
from pathlib import Path
import json,html,re,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import median_exploration_20261115 as ex

def paired(a,b,seed=20261115):
    out={};rng=np.random.default_rng(seed)
    for m in (1,4,64):
        aa=[r for r in a if r['count_multiplier']==m];bb=[r for r in b if r['count_multiplier']==m]
        assert [r['scene_id'] for r in aa]==[r['scene_id'] for r in bb]
        keys=sorted({r['background_id'] for r in aa});ix={k:i for i,k in enumerate(keys)}
        x=np.zeros((len(keys),3));y=x.copy()
        for arr,rows in ((x,aa),(y,bb)):
            for r in rows:arr[ix[r['background_id']]] += [r['tp'],r['fp'],r['fn']]
        draws=rng.integers(0,len(keys),(2000,len(keys)));xs=x[draws].sum(1);ys=y[draws].sum(1)
        f=lambda z:2*z[...,0]/np.maximum(2*z[...,0]+z[...,1]+z[...,2],1)
        delta=f(xs)-f(ys)
        out[str(m)]=dict(delta=float(f(x.sum(0))-f(y.sum(0))),ci95=np.quantile(delta,[.025,.975]).tolist(),backgrounds=len(keys))
    return out

NAMES={'baseline':'Published corrected median','original_score_restructure':'Original score + new boxes','classical':'Selected classical revision','per_count':'Count-specific classical revision','position_prior':'Revision + three-position prior','cnn':'Frozen CNN'}

def simple_markdown(text):
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',s)
        s=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',s)
        s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
        return s
    out=[];table=False;listing=False;code=False
    for line in text.splitlines():
        if line.startswith('```'):
            out.append('</code></pre>' if code else '<pre><code>');code=not code;continue
        if code:out.append(html.escape(line)+'\n');continue
        if line.startswith('|'):
            if not table:out.append('<table>');table=True;first=True
            if re.fullmatch(r'[| :\-]+',line):continue
            tag='th' if first else 'td';first=False
            out.append('<tr>'+''.join(f'<{tag}>'+inline(v.strip())+f'</{tag}>' for v in line.strip('|').split('|'))+'</tr>');continue
        if table:out.append('</table>');table=False
        if line.startswith('- '):
            if not listing:out.append('<ul>');listing=True
            out.append('<li>'+inline(line[2:])+'</li>');continue
        if listing:out.append('</ul>');listing=False
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
        elif line:out.append('<p>'+inline(line)+'</p>')
    if table:out.append('</table>')
    if listing:out.append('</ul>')
    return '\n'.join(out)

def description(op):
    shape='original shift/padding' if op['shape']=='original' else f"width-derived height (aspect {op['aspect']:g})"
    return f"{op['method']}; threshold {op['threshold']:g}; closing {op['closing']}; minimum area {op['area']}; {shape}; merge={op['merge']}; three-position snapping={op['snap']}"

def failure_breakdown(results):
    out={}
    for name,result in results.items():
        out[name]={}
        for count in (1,4,64):
            rows=[r for r in result['rows'] if r['count_multiplier']==count and r['truth_boxes']]
            buckets=dict(correct=0,overlap_below_iou_threshold=0,no_overlap=0)
            sizes={k:dict(tp=0,total=0) for k in ('below_0.3_mm','0.3_to_0.6_mm','at_least_0.6_mm')}
            for r in rows:
                best=max((ex.metrics.circular_iou(p,t,5024) for p in r['predicted_boxes'] for t in r['truth_boxes']),default=0.)
                buckets['correct' if best>=.5 else 'overlap_below_iou_threshold' if best>0 else 'no_overlap']+=1
                k='below_0.3_mm' if r['diameter_mm']<.3 else '0.3_to_0.6_mm' if r['diameter_mm']<.6 else 'at_least_0.6_mm'
                sizes[k]['tp']+=r['tp'];sizes[k]['total']+=1
            out[name][str(count)]=dict(**buckets,by_diameter=sizes)
    return out

def rect(ax,box,centre,color,label=None):
    x,y,r,b=box;mid=(x+r)/2
    delta=(mid-centre+2512)%5024-2512
    ax.add_patch(plt.Rectangle((delta-(r-x)/2,y),r-x,b-y,fill=False,edgecolor=color,lw=1.7,label=label))

def examples(results):
    metas=ex.read(ex.OUT/'datasets/test/index.json');lookup={r['scene_id']:r for r in metas}
    baseline={r['scene_id']:r for r in results['baseline']['rows']}
    revised={r['scene_id']:r for r in results['classical']['rows']}
    candidates=[r for r in baseline.values() if r['count_multiplier']==64 and r['truth_boxes']]
    improved=[r for r in candidates if r['tp']==0 and revised[r['scene_id']]['tp']==1]
    remaining=[r for r in candidates if revised[r['scene_id']]['tp']==0]
    picked=[]
    for group,key,rev in [(improved,lambda r:r['diameter_mm'],True),(improved,lambda r:r['diameter_mm'],False),(remaining,lambda r:r['diameter_mm'],True)]:
        if group:
            chosen=sorted(group,key=key,reverse=rev)[0]
            if chosen['scene_id'] not in [r['scene_id'] for r in picked]:picked.append(chosen)
    if not picked:picked=candidates[:2]
    cal=ex.read(ex.OUT/'calibration.json');op=ex.read(ex.OUT/'frozen.json')['policies']['classical']['op']
    fig,axes=plt.subplots(len(picked),3,figsize=(12,3.6*len(picked)),squeeze=False)
    records=[]
    for row,aa in zip(picked,axes):
        meta=lookup[row['scene_id']];raw=np.load(meta['raw_path']);truth=row['truth_boxes'][0]
        centre=(truth[0]+truth[2])/2;cols=(np.arange(-28,29)+int(round(centre)))%5024
        maps=ex.score_maps(raw,cal,{'original_static',op['method']})
        arrays=[ex.base.normalize(raw,ex.NORM)[16:144],maps['original_static'],maps[op['method']]]
        titles=['TOPO with geometry truth','Published median score and boxes','Revised score and boxes']
        yy=max(0,int((truth[1]+truth[3])/2)-24);yb=min(128,yy+48)
        for i,(ax,array,title) in enumerate(zip(aa,arrays,titles)):
            crop=array[yy:yb,cols]
            ax.imshow(crop,origin='lower',extent=(-28.5,28.5,yy,yb),cmap='gray' if i==0 else 'magma',aspect='equal',vmin=0,vmax=1 if i==0 else 24)
            rect(ax,truth,centre,'lime')
            for b in (row['predicted_boxes'] if i==1 else revised[row['scene_id']]['predicted_boxes'] if i==2 else []):rect(ax,b,centre,'cyan')
            ax.set_xlim(-28.5,28.5);ax.set_ylim(yy,yb);ax.set_title(title,fontsize=10);ax.set_xlabel('Angular offset (pixels)')
        aa[0].set_ylabel(f"D={row['diameter_mm']:.3f} mm\nScored radial row")
        records.append(dict(scene_id=row['scene_id'],diameter_mm=row['diameter_mm'],baseline_tp=row['tp'],revised_tp=revised[row['scene_id']]['tp']))
    fig.suptitle('Selected 64× examples: green = true opening; cyan = predicted box\nScores share 0–24 scale; selected extremes and remaining failure',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ex.REPORT/'examples_64x.png',dpi=180);plt.close(fig)
    return records

def reference_example(ablations):
    sid='val-031-open_intrinsic-c64-r0'
    meta=next(r for r in ex.read(ex.OLD/'datasets/val/index.json') if r['scene_id']==sid)
    raw=np.load(meta['raw_path']);maps=ex.score_maps(raw,ex.read(ex.OUT/'calibration.json'),('original_adaptive','short_adaptive'))
    long=next(r for r in ablations['results']['long_references']['rows'] if r['scene_id']==sid)
    short=next(r for r in ablations['results']['selected']['rows'] if r['scene_id']==sid)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for aa,(centre,half,yrange,label) in zip(axes,[(3388,90,(0,90),'Remote background false response'),(4399,28,(20,70),'Actual pore opening')]):
        cols=np.arange(centre-half,centre+half)%5024
        for j,(ax,im,title) in enumerate(zip(aa,[ex.base.normalize(raw,ex.NORM)[16:144],maps['original_adaptive'],maps['short_adaptive']],['Input TOPO','Distant-reference score','Closer-reference score'])):
            crop=im[yrange[0]:yrange[1],cols]
            ax.imshow(crop,origin='lower',extent=(-half,half,*yrange),aspect='auto',cmap='gray' if j==0 else 'magma',vmin=0,vmax=1 if j==0 else 24)
            if j:
                for b in (long if j==1 else short)['predicted_boxes']:rect(ax,b,centre,'cyan')
            for b in long['truth_boxes']:rect(ax,b,centre,'lime')
            ax.set_xlim(-half,half);ax.set_ylim(*yrange);ax.set_title(title,fontsize=10);ax.set_xlabel('Angular offset (pixels)')
        aa[0].set_ylabel(label+'\nScored radial row')
    fig.suptitle('Validation example: only reference offsets differ between the two score pipelines\nScore panels share 0–24 scale; green = opening; cyan = resulting boxes',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ex.REPORT/'reference_spacing_example.png',dpi=180);plt.close(fig)

def renderer_example():
    records=ex.read(ex.OUT/'renderer_sensitivity/000/index.json')
    modes=('original','B0971','bounded_yield');panels=[]
    cal=ex.read(ex.OUT/'calibration.json')
    op=ex.read(ex.OUT/'frozen.json')['policies']['classical']['op']
    for mode in modes:
        r=next(r for r in records if r['mode']==mode and r['source_class']=='open_intrinsic')
        clean=next(r for r in records if r['mode']==mode and r['source_class']=='clean')
        raw=np.load(r['raw_path']);truth=ex.truth_of(r)[0]
        centre=(truth[0]+truth[2])/2;cols=(np.arange(-28,29)+int(round(centre)))%5024
        low=max(0,int((truth[1]+truth[3])/2)-24);high=min(128,low+48)
        images=[ex.normalized(raw)[16:144],
                (ex.normalized(np.load(r['quiet_path']))-ex.normalized(np.load(clean['quiet_path'])))[16:144],
                ex.score_maps(raw,cal,{op['method']})[op['method']]]
        panels.append(dict(images=[a[low:high,cols] for a in images],truth=truth,centre=centre,low=low,high=high))
    limit=max(float(abs(p['images'][1]).max()) for p in panels)
    fig,axes=plt.subplots(3,3,figsize=(11,9))
    for j,(mode,p) in enumerate(zip(modes,panels)):
        for i in range(3):
            ax=axes[i,j]
            a=ax.imshow(p['images'][i],origin='lower',extent=(-28.5,28.5,p['low'],p['high']),aspect='equal',
                        cmap=('gray','coolwarm','magma')[i],vmin=(-limit if i==1 else 0),vmax=(1,limit,24)[i])
            rect(ax,p['truth'],p['centre'],'lime');ax.set_xlabel('Angular offset (pixels)')
            if i==0:ax.set_title(mode)
            if j==0:ax.set_ylabel(('Preprocessed TOPO','Quiet pore minus clean','Frozen revised score')[i]+'\nScored radial row')
            if j==2:fig.colorbar(a,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Matched renderer example: background 000, intrinsic pore, 64×\nShared scales within each row; green = unchanged true opening',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ex.REPORT/'renderer_example.png',dpi=180);plt.close(fig)

def main():
    ex.verify(True);ex.REPORT.mkdir(exist_ok=True)
    results=ex.read(ex.OUT/'confirmation_median.json');results['cnn']=ex.read(ex.OUT/'confirmation_cnn.json')
    sel=ex.read(ex.OUT/'selection.json');diag=ex.read(ex.OUT/'diagnostic.json');freeze=ex.read(ex.OUT/'frozen.json')
    ablations=ex.read(ex.OUT/'validation_ablations.json');reference=ex.read(ex.OUT/'reference_bias_diagnostic.json')
    comparisons={name:paired(v['rows'],results['baseline']['rows']) for name,v in results.items() if name!='baseline'}
    versus_cnn=paired(results['classical']['rows'],results['cnn']['rows'])
    final=dict(results={k:{q:v[q] for q in ('by_count','macro_f1')} for k,v in results.items()},paired_vs_baseline=comparisons,classical_vs_cnn=versus_cnn,
               validation_ablations={k:{q:v[q] for q in ('by_count','macro_f1')} for k,v in ablations['results'].items()},reference_diagnostic={k:v for k,v in reference.items() if k!='rows'},
               scope='48 new independent backgrounds; paired 2000-background-bootstrap intervals conditional on frozen methods. Selection uncertainty and simulator error excluded.')
    final['failure_breakdown']=failure_breakdown(results)
    fig,ax=plt.subplots(figsize=(9,5))
    for name in NAMES:
        vals=[results[name]['by_count'][str(m)]['f1'] for m in (1,4,64)]
        ax.plot(range(3),vals,'o-',label=NAMES[name])
    ax.set_xticks(range(3),['1×','4×','64×']);ax.set_ylim(0,1);ax.set_ylabel('Detection F1, box IoU ≥ 0.5');ax.set_xlabel('Effective count level');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right');ax.set_title('Fresh confirmation: 48 backgrounds, 864 scenes')
    fig.tight_layout();fig.savefig(ex.REPORT/'confirmation_f1.png',dpi=180);plt.close(fig)
    final['examples']=examples(results);reference_example(ablations)
    physics_path=ex.OUT/'renderer_sensitivity/results.json'
    physics=ex.read(physics_path) if physics_path.exists() else None
    if physics is not None:
        final['renderer_sensitivity']={m:{k:v['by_count']['64'] for k,v in methods.items()} for m,methods in physics.items()}
        renderer_example()
    stages=diag['stages'];final['diagnostic_stages']=dict(pore_acquisitions=len(stages),no_pixel_above_threshold_inside=sum(s['above_threshold_pixels_inside']==0 for s in stages),overlap_before_area=sum(s['overlaps_before_area'] for s in stages),overlap_after_area=sum(s['overlaps_after_area'] for s in stages),matched_before_correction=sum(s['matches_before_correction'] for s in stages),matched_after_correction=sum(s['matches_after_correction'] for s in stages))
    ex.write(ex.OUT/'final_summary.json',final);ex.write(ex.REPORT/'results.json',final)
    b64=results['baseline']['by_count']['64'];r64=results['classical']['by_count']['64'];dc=comparisons['classical']['64']
    lines=['# Median improvement experiment','', '15 November 2026','',
      '## Principal findings','',
      f"On fresh confirmation at 64×, the selected non-snapping classical revision achieved F1 {r64['f1']:.3f}, compared with {b64['f1']:.3f} for the published corrected median. The paired difference was {dc['delta']:+.3f}, with a 95% background-bootstrap interval [{dc['ci95'][0]:+.3f}, {dc['ci95'][1]:+.3f}]. Correctly localised acquisitions changed from {b64['tp']} to {r64['tp']} of 192.", '',
      'The evidence supports a 64× improvement within this synthetic experiment, not an improvement at every noise level. The 1× and 4× difference intervals include zero. At 1×, the revision increases precision while reducing recall: correct localisations fall from 76 to 49, while false regions fall from 316 to 48. At 4×, correct localisations are effectively unchanged (69 versus 68). The CNN remains stronger at all three levels.','',
      'The primary revision’s 64× validation F1 was 0.698, but fresh confirmation F1 was 0.529. Its mean F1 fell from 0.610 to 0.426. This development-to-confirmation gap limits confidence in the size of the validation gain. Selection optimism and sampling variation are plausible contributors; their separate effects have not been established. A 12-scene historical CNN replay reproduced archived boxes and metrics exactly, which checks that the new evaluation path matches the existing one.','',
      f"The validation diagnostic retained {diag['results']['measured']['by_count']['64']['tp']} correct localisations with measured input and {diag['results']['quiet']['by_count']['64']['tp']} with quiet input, using the same detector settings. Quiet input removes the acquisition stage, including noise and acquisition gain/contamination effects; it is not an isolated Poisson-noise ablation. This idealised input did not resolve the localisation bottleneck in that subset. The paired-clean oracle also did not make direct thresholded contrast equivalent to the opening mask under the tested extraction rules.", '',
      f"In the 64-pore validation diagnostic, the area filter reduced overlapping detections from {final['diagnostic_stages']['overlap_before_area']} to {final['diagnostic_stages']['overlap_after_area']}; the existing box correction changed correctly localised detections from {final['diagnostic_stages']['matched_before_correction']} to {final['diagnostic_stages']['matched_after_correction']}. These isolate particular processing stages, while the selected-policy comparison may change several settings together.", '',
      '## What was done','',
      'Compared six residual-score constructions and 1,872 bounded configurations using the original 96 training and 32 validation backgrounds. New static calibrations used only clean training images. The input blur remained sigma 1 pixel. Candidate settings and the existing CNN were frozen before generating 48 new confirmation backgrounds (864 scenes; 192 pore and 96 clean scenes per count level). The image generator and historical datasets were preserved.','',
      'The published corrected median was replayed exactly on validation: mean per-count F1 0.39972199985. This checks the scoring and baseline implementation before comparing changes.','',
      '## Fresh confirmation results','',
      '| Method | 1× F1 | 4× F1 | 64× F1 | Mean F1 | Clean alarms at 64× |','|---|---:|---:|---:|---:|---:|']
    for k,n in NAMES.items():
        v=results[k];b=v['by_count'];lines.append(f"| {n} | {b['1']['f1']:.3f} | {b['4']['f1']:.3f} | {b['64']['f1']:.3f} | {v['macro_f1']:.3f} | {b['64']['clean_alarms']}/{b['64']['clean_scenes']} |")
    lines+=['','![Fresh confirmation F1](confirmation_f1.png)','',
      'All rows use the same fresh scenes. The count-specific policy requires the acquisition count level to be known. The three-position method uses the simulator’s restricted radial-location prior and is not evidence of general localisation on arbitrary pore positions.','',
      '### Paired improvement over the published median','',
      '| Method | ΔF1 at 1× [95% interval] | ΔF1 at 4× [95% interval] | ΔF1 at 64× [95% interval] |','|---|---|---|---|']
    for k,d in comparisons.items():
        fmt=lambda m:f"{d[m]['delta']:+.3f} [{d[m]['ci95'][0]:+.3f}, {d[m]['ci95'][1]:+.3f}]"
        lines.append(f"| {NAMES[k]} | {fmt('1')} | {fmt('4')} | {fmt('64')} |")
    lines+=['','### What changed at 64×','',
      '| Method | TP | FP | FN | Precision | Recall | Missed with overlap | Missed without overlap |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('baseline','classical','cnn'):
        b=results[k]['by_count']['64'];f=final['failure_breakdown'][k]['64']
        lines.append(f"| {NAMES[k]} | {b['tp']} | {b['fp']} | {b['fn']} | {b['precision']:.3f} | {b['recall']:.3f} | {f['overlap_below_iou_threshold']} | {f['no_overlap']} |")
    lines+=['','A missed acquisition with overlap has a predicted box touching the true opening, but no box reaches IoU 0.5. A missed acquisition without overlap may have predictions elsewhere. FP counts include unmatched regions in defective scenes as well as clean-scene alarms.','',
      '| Pore diameter | Baseline correct / acquisitions | Revised correct / acquisitions | CNN correct / acquisitions |','|---|---:|---:|---:|']
    for size in ('below_0.3_mm','0.3_to_0.6_mm','at_least_0.6_mm'):
        cells=[]
        for k in ('baseline','classical','cnn'):
            z=final['failure_breakdown'][k]['64']['by_diameter'][size];cells.append(f"{z['tp']} / {z['total']}")
        lines.append('| '+size.replace('_',' ')+' | '+' | '.join(cells)+' |')
    lines+=['','Diameter groups are descriptive and include two acquisition repeats per geometry. These are recall counts, not independently sampled pores or group-specific F1 scores.','']
    lines+=['The revision recovers 18 additional medium-pore acquisitions (42 to 60), two additional large-pore acquisitions (12 to 14), and no net additional small-pore acquisitions (18 in both cases). Large-pore recall remains 14/60, compared with 52/60 for the CNN. Across all sizes, 64 of the revision’s 100 missed pore acquisitions still have an overlapping box below the IoU threshold. Thus opening extent remains a major limitation even after the improvement.','']
    lines+=['','Intervals use 2,000 paired background bootstrap resamples at each count level. Related surfaces and repeats are resampled together, and the compared methods use the same background draws. They condition on the selected models, omit simulator error and selection variability, and are not adjusted for multiple exploratory comparisons.','',
      '### Operating points','']
    for k in ('classical','original_score_restructure','position_prior'):
        v=freeze['policies'][k];lines.append(f"- **{NAMES[k]}:** {description(v['op'])}. Validation mean F1 {v['macro_f1']:.3f}.")
    for m,v in freeze['policies']['per_count'].items():lines.append(f"- **Count-specific {m}×:** {description(v['op'])}.")
    lines+=['','### Exact selected detector','',
      'Blur the raw TOPO array with Gaussian sigma 1 pixel, then apply the unchanged training-derived intensity normalisation. At each row, form a median reference from angular offsets ±32, ±48, ±64 and ±80 pixels (1.6–4.0 mm at the campaign sampling). The previous reference used ±64, ±128, ±192 and ±256 pixels (3.2–12.8 mm).','',
      'Subtract this reference from the normalised image. For each row, estimate the residual median and 1.4826 times its median absolute deviation across angular columns, with a scale floor of 0.003. The score is the absolute centred residual divided by this scale. Threshold at 16, close with a 3 × 3 kernel, and retain connected components of at least 3 pixels. Angular boundaries are periodic.','',
      'Merge qualifying vertically separated components that overlap by at least half the narrower angular width, have a gap no greater than 8 pixels and combined extent no greater than 40 pixels. The implementation permits at most six merges. Shift the centre 2 rows towards the ring centre, expand angular width by 2 pixels, and infer height as max(3, expanded width / 1.6). Integer rounding and clipping follow the saved implementation. No snapping to the three nominal pore positions is used in the primary revision.','',
      'The inference entry point `WeldDefectDetection/apply_revised_median_20261115.py` accepts raw registered 160 × 5024 .npy arrays and returns cyclic boxes in both scored and raw coordinates. It does not access truth masks. Twelve prediction replays agreed exactly with the frozen experiment. Its default uses the single shared policy; `--count` optionally selects the frozen count-specific policy.','']
    lines+=['','### Performance by score construction on validation','',
      'Each row below is its best non-snapping region configuration; differences can include separately selected postprocessing and are not isolated causal effects.','',
      '| Score | Best mean validation F1 |','|---|---:|']
    for k,v in sel['best_by_score'].items():lines.append(f"| {k} | {v['macro_f1']:.3f} |")
    lines+=['','### Controlled removals from the selected revision','',
      'Validation-only diagnostics hold the other selected settings fixed. These are descriptive checks after selection, not additional independent confirmation results.','',
      '| Change | 1× F1 | 4× F1 | 64× F1 | 64× TP / FP / FN |','|---|---:|---:|---:|---|']
    for k,v in ablations['results'].items():
        b=v['by_count'];z=b['64'];lines.append(f"| {k} | {b['1']['f1']:.3f} | {b['4']['f1']:.3f} | {z['f1']:.3f} | {z['tp']} / {z['fp']} / {z['fn']} |")
    lines+=['',
      'Restoring the distant reference offsets has the largest adverse effect in this combination. Static calibration mainly damages high-noise precision. Width-derived height and a smaller minimum area recover additional localisations. Disabling fragment merging has a smaller effect. These interactions do not justify attributing the total improvement to one operation.','',
      '### Direct clean-reference check','',
      f"For 64 paired validation pore geometries, the median short-to-long ratio of noise-free clean-reference RMS error inside the opening was {reference['median_paired_short_to_long_bias_ratio']:.3f}; the short reference had lower error in {reference['short_lower_bias_geometries']} of 64 geometries. The median ratio of long-to-short adaptive scale was {reference['median_long_to_short_adaptive_scale_ratio']['measured']:.3f} on measured images. Thus neither universal reference-bias reduction nor uniform scale inflation explains the whole F1 change. Threshold support and component geometry also change. Pore contamination of the median reference inside the opening had zero median and zero 90th percentile in this diagnostic. This does not exclude effects elsewhere on the ring.", '']
    lines+=['![Reference-spacing example](reference_spacing_example.png)','',
      'This selected validation scene (val-031-open_intrinsic-c64-r0) illustrates both a remote background response and a missed local opening under distant references. All other detector settings are held fixed. The example illustrates the measured effect; it does not establish how frequently this particular appearance occurs.','']
    ghost=ex.read(ex.OUT/'reference_ghost_diagnostic.json')
    lines += [f"A separate quiet-image check isolated changes to the median reference at locations outside the opening where the direct pore-induced image change was below 1e-5. Neither reference scheme produced a retained ghost component at threshold 16, area 3 and closing 3 in these 64 geometries. Maximum isolated ghost scores were {ghost['summary']['original']['max_ghost_quantiles'][2]:.2f} for distant references and {ghost['summary']['short']['max_ghost_quantiles'][2]:.2f} for closer references. This does not support strong isolated pore ghosts as the main explanation, although subthreshold contamination can still combine with background residuals.", '']
    lines+=['','## Mechanism diagnostics','',
      'These use validation repeat 0 at 64×: 32 backgrounds, 64 pore acquisitions and 32 clean acquisitions. They were not used as fresh confirmation evidence. Paired clean subtraction requires an unavailable defect-free counterpart and is an oracle diagnostic, not a proposed deployable method. Its per-image maximum normalisation and looser minimum area also change the operating conditions.','',
      '| Diagnostic | TP | FP | FN | F1 |','|---|---:|---:|---:|---:|']
    for k,v in diag['results'].items():
        b=v['by_count']['64'];lines.append(f"| {k} | {b['tp']} | {b['fp']} | {b['fn']} | {b['f1']:.3f} |")
    lines+=['', 'Here `measured` and `quiet` use identical original-median calibration, threshold and box correction. Their difference isolates removal of the acquisition effects represented in the saved quiet image. `oracle` subtracts paired quiet clean and pore images after common preprocessing, then thresholds fractions of the difference maximum. Width reconstruction infers box height from angular width rather than equating the retained contrast region with the opening.','',
      '### Baseline stage counts','', '```json',json.dumps(final['diagnostic_stages'],indent=2),'```','',
      'Overlap counts only establish that a component touches the opening. They do not establish accurate extent. A score below threshold inside the mask does not exclude a displaced response outside it.','',
      '## Illustrative cases','', '![Selected cases](examples_64x.png)','',
      'Examples are selected as the largest and smallest improved pore, plus a large remaining failure where available. They illustrate behaviour rather than estimate its frequency. Exact scene identities are recorded in results.json.','']
    if physics is not None:
        lines+=['## Renderer sensitivity','',
          'Eight fixed new backgrounds were re-rendered at 64×, repeat 0 (16 pore and 8 clean acquisitions per variant), with identical geometry and acquisition settings. Original-mode replay was required to reproduce the saved images bit for bit. The same frozen detectors were used throughout, with no retuning.','',
          '- **B0971:** replace B = 0 with constant B = 0.971. This retains the known polar-law defect and extrapolates the scalar form beyond the checked CASINO tilt range.','- **Bounded yield:** use eta0 × [2/(1 + cos(alpha))]^(9/sqrt(Z)), preserving normal-incidence yield while replacing the early saturation of eta0/cos(alpha)^p. This is a sensitivity perturbation, not a calibrated physical correction.','',
          '| Method | Original 64× F1 | B = 0.971 F1 | Bounded-yield F1 |','|---|---:|---:|---:|']
        for k in NAMES:lines.append(f"| {NAMES[k]} | {physics['original'][k]['by_count']['64']['f1']:.3f} | {physics['B0971'][k]['by_count']['64']['f1']:.3f} | {physics['bounded_yield'][k]['by_count']['64']['f1']:.3f} |")
        lines+=['','![Matched renderer example](renderer_example.png)','',
          'The displayed background and pore were fixed for illustration rather than selected for the largest change. The middle row uses a paired defect-free image unavailable at inference.','',
          'These are small, paired distribution-shift diagnostics. The training normalisation, calibration, thresholds and CNN weights remain unchanged, so sensitivity can include intensity scaling as well as spatial contrast. They do not determine the best detector after retraining under another renderer, or which renderer is physically correct.','']
        lines+=['The selected revision falls from F1 0.500 to 0.385 with B = 0.971 and to 0.286 with the bounded yield. The original-score reconstruction variant rises from 0.348 to 0.583 under B = 0.971, so the response is method-dependent. Neither renderer change supplies a general performance fix. Choosing a rendering law because it makes detection easier would not establish physical correctness; physical calibration must precede a new training and confirmation campaign.','']
    lines+=['## Recommended decision','',
      'Retain the selected closer-reference median as an improved 64× baseline and report its trade-offs at lower counts. Preserve the CNN as the stronger detector in this experiment. Do not replace the renderer solely to maximise detector F1, and do not claim that the new median resolves the localisation problem.','',
      'The next high-value algorithm experiment is size-aware opening reconstruction on development data: estimate a pore region from the signed contrast lobes or fit candidate pore-response shapes at several sizes, rather than infer every height from a single aspect ratio. This is motivated by the 64 overlapping misses and low large-pore recall, but has not been tested here. Any subsequent method must be frozen before evaluation on another fresh set.','',
      'The radial-snapping variant reaches 64× F1 0.652 in this experiment, but uses the three allowed simulator positions. It is a useful restricted-prior result, not the general recommendation. The count-specific variant is also reported in full; its 4× F1 of 0.361 is below the baseline 0.398, which cautions against assuming that more specialised validation tuning transfers better.','']
    lines+=['## Implication for the thesis','',
      'Treat this as a separately documented follow-up study. Preserve the historical confirmation table and add the new paired comparison, which uses different backgrounds. If the stronger baseline is adopted in the main discussion, describe its reference spacing, adaptive scaling and shape-based box reconstruction explicitly. Conclusions about the original implementation should not be broadened to all median-based detectors.','',
      f"Suggested completed-study wording: An exploratory follow-up compared 1,872 classical detector configurations on the existing development split and froze the selected methods before generating 48 new confirmation backgrounds. At 64×, the selected median revision achieved detection F1 {r64['f1']:.3f}, compared with {b64['f1']:.3f} for the original corrected baseline on the same scenes. The improvement was {dc['delta']:.3f} (paired 95% background-bootstrap interval {dc['ci95'][0]:.3f} to {dc['ci95'][1]:.3f}). Validation diagnostics identified sensitivity to angular reference spacing, minimum region area and box reconstruction. Removing acquisition effects did not resolve the baseline localisation problem. These results establish that the original classical baseline was improvable within the specified synthetic distribution; they do not establish measured-image performance or physical validity of the renderer.", '',
      '## Limits and reproduction','',
      '- All results remain synthetic, with known ring coordinates, one pore per defective scene, two prescribed pore families and three radial positions.','- The width-to-height reconstruction uses a pore-shape prior even when radial snapping is disabled. It requires testing on more diverse shapes.','- Validation was reused from prior work. Fresh confirmation reduces direct test reuse but does not make the study a preregistered independent replication.','- No new CNN was trained. The unchanged selected model provides a matched reference on the new images.','- Historical files and source dependencies are checked by hash. The new experiment is isolated in its own output directory.','',
      'Scripts: `WeldDefectDetection/median_exploration_20261115.py`, `WeldDefectDetection/median_renderer_sensitivity_20261115.py`, and `WeldDefectDetection/report_median_exploration_20261115.py`. Run them with the project `.venv/Scripts/python.exe`. Protocol, frozen settings, per-scene predictions, complete validation trials, and bootstrap summaries are in `DataSimulation/data/output/median_exploration_20261115_v1`.','',
      'Main run order: `prepare`, `development`, `freeze`, `generate`, `confirm`, `cnn`; renderer sensitivity: `prepare`, `generate`, `evaluate`; then run the report script. Existing completed per-scene work is reused. Do not rerun `freeze` after confirmation generation.']
    text='\n'.join(lines)+'\n';(ex.REPORT/'REPORT.md').write_text(text,encoding='utf-8')
    # Standalone readable HTML using simple Markdown conversion when available.
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        body=simple_markdown(text)
    # Embed figures so the HTML remains readable when copied on its own.
    def embed_image(match):
        p=ex.REPORT/match.group(1)
        return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')+'"'
    body=re.sub(r'src="([^"]+\.png)"',embed_image,body)
    style='body{font:16px/1.55 Segoe UI,Arial,sans-serif;color:#172b3a;max-width:1150px;margin:36px auto;padding:0 24px}h1,h2,h3{color:#123f61}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{padding:9px;border:1px solid #cbd5df;text-align:left}th{background:#e9f0f6}img{max-width:100%}pre{background:#f2f5f7;padding:16px;overflow:auto}'
    (ex.REPORT/'REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Median improvement experiment</title><style>'+style+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
    print(json.dumps(final['results'],indent=2));print('REPORT',ex.REPORT/'REPORT.md')

if __name__=='__main__':main()
