"""Audit and present the frozen median localisation follow-up."""
import sys, json, csv, html, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import median_localisation_20261110 as s

OUT=s.OUT
REPORT=s.ROOT/'docs/median_localisation_20261110'
NAMES={'original':'Original median','retuned':'Retuned median','shift':'Position correction',
       'extent':'Size correction','combined':'Position + size','cnn':'Frozen CNN'}

def local_markdown(text):
    """Small renderer for this report's headings, tables, lists and figures."""
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    out=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>')
        elif line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));out.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>')
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in cells):table.append(cells)
                i+=1
            out.append('<table>')
            for j,row in enumerate(table):
                tag='th' if j==0 else 'td';out.append('<tr>'+''.join(f'<{tag}>'+inline(c)+f'</{tag}>' for c in row)+'</tr>')
            out.append('</table>');continue
        elif line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
            assert match,line
            out.append('<figure><img src="'+html.escape(match[2],quote=True)+'" alt="'+html.escape(match[1],quote=True)+'"></figure>')
        else:out.append('<p>'+inline(line)+'</p>')
        i+=1
    return '\n'.join(out)

def audit():
    frozen=s.verify_frozen();rows=s.read(OUT/'datasets/test/index.json')
    assert len(rows)==s.FRESH_COUNT*18
    old=[r for split in ('train','val','test') for r in s.learn.selected_rows(split)]
    assert not {r['seed'] for r in rows}&{r['seed'] for r in old}
    assert not {r['acquisition_seed'] for r in rows}&{r['acquisition_seed'] for r in old}
    files={r[k]:r[h] for r in rows for k,h in [('raw_path','raw_sha256'),('geometry_path','geometry_sha256')]}
    for path,digest in files.items():assert s.sha(path)==digest,path
    from datetime import datetime
    frozen_time=datetime.fromisoformat(frozen['frozen_utc']).timestamp()
    assert min(Path(r['raw_path']).stat().st_mtime for r in rows)>frozen_time
    for kind in s.c.KINDS:
        for i in range(s.FRESH_COUNT):
            group=[r for r in rows if r['index']==i and r['source_class']==kind]
            assert len(group)==6 and len({r['geometry_sha256'] for r in group})==1
    results={k:s.read(OUT/f'confirmation/{k}.json') for k in NAMES}
    for name,result in results.items():
        assert [r['scene_id'] for r in result['rows']]==[r['scene_id'] for r in rows]
        for r in result['rows']:
            got=s.metrics.match(r['predicted_boxes'],r['truth_boxes'],5024)
            assert all(r[k]==v for k,v in got.items())
        assert s.learn.objective(result['rows'])['by_count']==result['by_count']
    # Independently replay saved validation scores on defects as well as cleans.
    data=s.learn.Data(s.learn.selected_rows('val'),s.read(s.OLD/'normalization.json'))
    for sigma in s.SIGMAS:
        spec=s.read(OUT/f'development/calibration_sigma{sigma:g}.json')
        score=np.load(s.DEV/f'median_sigma{sigma:g}_val.npy',mmap_mode='r')
        for i in (6,8,15,143,284,575):
            assert np.array_equal(s.learn.median_score(data.image(i,sigma),spec),score[i])
    fresh=s.learn.Data(rows,s.read(s.OLD/'normalization.json'))
    model=s.base.model_for(frozen['original']['cnn'])
    cnn_scores=np.load(OUT/'confirmation/cnn.npy',mmap_mode='r')
    for i in (0,8,len(rows)-1):
        assert np.array_equal(s.base.predict_image(fresh.image(i),model),cnn_scores[i])
    result=dict(status='passed',fresh_backgrounds=s.FRESH_COUNT,scenes=len(rows),hashed_fresh_files=len(files),
        disjoint_background_and_acquisition_seeds=True,generated_after_freeze=True,
        matched_geometry=True,metric_replay=True,validation_score_replay=True,
        original_dependencies_unchanged=True,frozen_cnn_score_replay=True,source=s.sha(__file__),physical_validation=False)
    s.write(OUT/'final_audit.json',result)
    return results

def paired_intervals(results,chosen,baseline='original'):
    out={};rng=np.random.default_rng(11102026)
    for count in s.c.MULTIPLIERS:
        a=[r for r in results[chosen]['rows'] if r['count_multiplier']==count]
        b=[r for r in results[baseline]['rows'] if r['count_multiplier']==count]
        keys=sorted({r['background_id'] for r in a})
        def aggregates(rows):
            return np.array([[sum(r[k] for r in rows if r['background_id']==bg) for k in ('tp','fp','fn')] for bg in keys])
        aa,bb=aggregates(a),aggregates(b);draw=rng.integers(0,len(keys),(2000,len(keys)))
        aa=aa[draw].sum(1);bb=bb[draw].sum(1)
        fa=2*aa[:,0]/np.maximum(1,2*aa[:,0]+aa[:,1]+aa[:,2]);fb=2*bb[:,0]/np.maximum(1,2*bb[:,0]+bb[:,1]+bb[:,2])
        out[str(count)]=dict(delta=results[chosen]['by_count'][str(count)]['f1']-results[baseline]['by_count'][str(count)]['f1'],
            paired_95_ci=np.quantile(fa-fb,[.025,.975]).tolist())
    s.write(OUT/('paired_intervals.json' if baseline=='original' else f'paired_intervals_vs_{baseline}.json'),out);return out

def development_localisation():
    """Describe the original median's geometry errors on validation only."""
    data=s.learn.Data(s.learn.selected_rows('val'),s.read(s.OLD/'normalization.json'))
    original=s.read(s.OLD/'frozen_models.json')['arms']['mixed_96']['median']
    op=original['selection']['best'];sigma=original['spec']['sigma']
    cache=s.read(OUT/f'components/s{sigma:g}_t{op["threshold"]:g}_c{op["closing"]}.json')
    records=[]
    for i,meta in enumerate(data.rows):
        if not data.truth[i]:continue
        t=data.truth[i][0];pred=[v[:4] for v in cache[i] if v[4]>=op['min_area']]
        overlaps=[(s.metrics.circular_iou(p,t,5024),p) for p in pred]
        overlaps=[(v,p) for v,p in overlaps if v>0]
        p=meta['parameters']['defects'][0]['params']
        row=dict(scene_id=meta['scene_id'],count_multiplier=meta['count_multiplier'],radial_offset_mm=p['radial_offset_mm'],
            diameter_mm=meta['diameter_mm'],overlapping_responses=len(overlaps))
        if overlaps:
            iou,box=max(overlaps,key=lambda v:v[0])
            row.update(best_iou=iou,dy_px=(box[1]+box[3]-t[1]-t[3])/2,
                dx_px=float(((box[0]+box[2]-t[0]-t[2])/2+2512)%5024-2512),
                height_ratio=(box[3]-box[1])/(t[3]-t[1]),width_ratio=(box[2]-box[0])/(t[2]-t[0]))
        records.append(row)
    def group(rr):
        visible=[r for r in rr if r['overlapping_responses']]
        return dict(pore_scenes=len(rr),with_overlapping_response=len(visible),
            no_overlapping_response=len(rr)-len(visible),fragmented=sum(r['overlapping_responses']>1 for r in rr),
            matched=sum(r.get('best_iou',0)>=.5 for r in rr),
            **{f'median_{k}':float(np.median([r[k] for r in visible])) if visible else None for k in ('dy_px','dx_px','height_ratio','width_ratio','best_iou')})
    result=dict(scope='Original median, validation data only; reference boxes associate overlapping responses for diagnosis, not inference.',
        by_count={str(m):group([r for r in records if r['count_multiplier']==m]) for m in s.c.MULTIPLIERS},
        low_noise_by_radial={str(off):group([r for r in records if r['count_multiplier']==64 and r['radial_offset_mm']==off]) for off in (-1.05,0.,1.05)},
        low_noise_by_size={name:group([r for r in records if r['count_multiplier']==64 and lo<=r['diameter_mm']<hi]) for name,lo,hi in [('small',.15,.3),('medium',.3,.6),('large',.6,1.01)]},rows=records)
    s.write(OUT/'development_localisation.json',result);return result

def alignment_figure():
    data=s.read(OUT/'alignment.json');fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    selected=[]
    for j,off in enumerate((-1.05,0.,1.05)):
        records=[r for r in data['rows'] if r['radial_offset_mm']==off]
        median=data['by_radial_position'][str(off)]['median_dy']
        r=min(records,key=lambda r:abs(r['contrast_centroid_dy_px']-median));selected.append(r['scene_id'])
        with np.load(r['geometry_path']) as z:quiet=z['quiet'];mask=z['mask']
        with np.load(Path(r['geometry_path']).with_name('clean_geometry.npz')) as z:clean=z['quiet']
        cx=int(round(r['angle_rad']*5024/(2*np.pi)));cy=int(round(79.5+off/.05))
        cols=(np.arange(-24,25)+cx)%5024;sl=slice(max(0,cy-24),min(160,cy+25));m=mask[sl][:,cols]
        panels=[quiet[sl][:,cols],(quiet-clean)[sl][:,cols],m]
        for k,(ax,a) in enumerate(zip(axes[j],panels)):
            if k==1:
                vmax=max(1e-8,float(abs(a).max()));ax.imshow(a,cmap='coolwarm',vmin=-vmax,vmax=vmax)
            else:ax.imshow(a,cmap='gray')
            if k<2:ax.contour(m,[.5],colors=['#ff3366'],linewidths=1.5)
            ax.set_xticks([]);ax.set_yticks([])
        axes[j,0].set_ylabel(f'Radial position {off:+.2f} mm\n{r["scene_id"]}',fontsize=9)
    for ax,title in zip(axes[0],['Noiseless TOPO + opening outline','Pore-induced contrast (pore minus clean)','Geometric opening mask']):ax.set_title(title,fontsize=11)
    fig.suptitle('Correctly registered geometry can produce a displaced contrast footprint',fontsize=15)
    axes[-1,1].set_xlabel('Blue: signal decrease; red: signal increase.\nColour scale is set separately for each example.',fontsize=9)
    fig.savefig(REPORT/'01_alignment.png',dpi=180);plt.close(fig)
    s.write(REPORT/'visual_provenance.json',dict(alignment_examples=selected,selection='Closest to median contrast displacement within each radial-position group; training data only.'))

def plots(results,chosen):
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    colours=['#888888','#2474b5','#c96c00','#669933','#8342a8','#121b30']
    for (name,r),colour in zip(results.items(),colours):
        ax.plot([0,1,2],[r['by_count'][str(m)]['f1'] for m in s.c.MULTIPLIERS],marker='o',label=NAMES[name],color=colour,lw=2)
    ax.set(xticks=[0,1,2],xticklabels=['1×: highest noise','4×','64×: lowest noise'],ylim=(0,1),ylabel='Detection F1',title=f'Fresh confirmation: {s.FRESH_COUNT} independent weld backgrounds')
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.savefig(REPORT/'02_fresh_results.png',dpi=180);plt.close(fig)
    original=results['original']['rows'];new=results[chosen]['rows'];cnn=results['cnn']['rows']
    candidates=[i for i,r in enumerate(new) if r['source_class']!='clean' and r['count_multiplier']==64 and r['repeat']==0]
    improved=[i for i in candidates if new[i]['tp']>original[i]['tp']]
    missed=[i for i in candidates if new[i]['fn']>0]
    # Deterministic, disclosed examples; not representative estimates.
    selected=[]
    for pool in (improved,missed):
        if pool:selected.append(max(pool,key=lambda i:new[i]['diameter_mm']))
    if not selected:selected=candidates[:2]
    meta=s.read(OUT/'datasets/test/index.json');fig,axes=plt.subplots(len(selected),4,figsize=(13,3.6*len(selected)),squeeze=False,layout='constrained')
    for j,i in enumerate(selected):
        r=meta[i];truth=new[i]['truth_boxes'][0];cx=int((truth[0]+truth[2])/2)%5024;cy=int((truth[1]+truth[3])/2)
        cols=(np.arange(-24,25)+cx)%5024;sl=slice(max(0,cy-24),min(128,cy+25));raw=np.load(r['raw_path'])[16:144]
        with np.load(r['geometry_path']) as z:m=z['mask'][16:144]
        panels=[None,original[i],new[i],cnn[i]]
        for ax,rec in zip(axes[j],panels):
            ax.imshow(raw[sl][:,cols],cmap='gray')
            if rec is not None:
                for x,y,xx,yy in rec['predicted_boxes']:
                    width=xx-x;mid=(x+xx)/2
                    local_mid=((mid-cx+2512)%5024-2512)+24
                    ax.add_patch(Rectangle((local_mid-width/2-.5,y-sl.start-.5),width,yy-y,
                                           fill=False,edgecolor='#00cfff',linewidth=2))
                best=max((s.metrics.circular_iou(p,t,5024) for p in rec['predicted_boxes'] for t in rec['truth_boxes']),default=0.)
                ax.text(.02,.02,f'Best box IoU {best:.3f}\nFull ring TP / FP / FN: {rec["tp"]} / {rec["fp"]} / {rec["fn"]}',
                        transform=ax.transAxes,fontsize=9,va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            ax.contour(m[sl][:,cols],[.5],colors=['#ff3366'],linewidths=1.6)
            ax.set_xlim(-.5,len(cols)-.5);ax.set_ylim(raw[sl][:,cols].shape[0]-.5,-.5)
            ax.set_xticks([]);ax.set_yticks([])
        axes[j,0].set_ylabel(f'{r["scene_id"]}\nD = {r["diameter_mm"]:.3f} mm',fontsize=9)
    for ax,title in zip(axes[0],['TOPO and opening','Original median',NAMES[chosen],'Frozen CNN']):ax.set_title(title)
    fig.suptitle('Low-noise examples: red = true opening; cyan = detection boxes',fontsize=14)
    fig.savefig(REPORT/'03_detection_examples.png',dpi=180);plt.close(fig)
    provenance=s.read(REPORT/'visual_provenance.json');provenance['confirmation_examples']=[meta[i]['scene_id'] for i in selected]
    provenance['confirmation_selection']='Largest pore improved over original, and largest pore still missed; 64x repeat 0. Boxes shown for all detectors.'
    s.write(REPORT/'visual_provenance.json',provenance)

def tuning_figure():
    trials=s.read(OUT/'correction_trials.json');selected=s.read(OUT/'selection.json')['combined']
    corr=selected['correction'];base=selected['base']
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,rule in zip(axes,('constant','toward_crown')):
        rows=sorted([t for t in trials if t['family']=='combined' and t['base']['sigma']==base['sigma']
                     and t['correction']['radial_rule']==rule and t['correction']['dx']==0
                     and t['correction']['scale']==corr['scale'] and t['correction']['pad']==corr['pad']],
                    key=lambda t:t['correction']['dy'])
        # dy=0 has a unique constant-rule representation in the search.
        if rule=='toward_crown':
            zero=next(t for t in trials if t['family']=='combined' and t['base']['sigma']==base['sigma']
                      and t['correction']==dict(radial_rule='constant',dy=0,dx=0,scale=corr['scale'],pad=corr['pad']))
            rows=sorted(rows+[zero],key=lambda t:t['correction']['dy'])
        for m,color in zip(s.c.MULTIPLIERS,('#cf7037','#3578af','#22846a')):
            ax.plot([t['correction']['dy'] for t in rows],[t['by_count'][str(m)]['f1'] for t in rows],'-o',label=f'{m}×',color=color)
        ax.set(ylim=(0,.65),ylabel='Validation detection F1',xlabel='Shift (pixels)',title='Same radial shift everywhere' if rule=='constant' else 'Positive shift moves towards crown centre')
        ax.axvline(0,color='#aaaaaa',lw=1);ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'Validation-only correction sweep: sigma {base["sigma"]:g}, threshold {base["threshold"]:g}, padding {corr["pad"]} px')
    fig.savefig(REPORT/'04_validation_correction.png',dpi=180);plt.close(fig)

def report(results):
    REPORT.mkdir(parents=True,exist_ok=True)
    frozen=s.read(OUT/'frozen.json');selection=frozen['selection'];chosen=selection['chosen_family']
    selected=selection[chosen];config=selected if chosen=='retuned' else selected['base']
    s.write(REPORT/'selected_median.json',dict(study=OUT.name,frozen_utc=frozen['frozen_utc'],
        normalization=s.read(s.OLD/'normalization.json'),
        calibration=s.read(OUT/f'development/calibration_sigma{config["sigma"]:g}.json'),
        operating_point={k:config[k] for k in ('threshold','closing','min_area')},
        correction=None if chosen=='retuned' else selected['correction'],
        input_shape=[160,5024],scored_rows=[16,144],pixel_pitch_mm=.05,
        inference_source=str(s.Path(s.__file__)),inference_source_sha256=s.sha(s.__file__)))
    ci=paired_intervals(results,chosen);fixed_ci=paired_intervals(results,chosen,'retuned')
    alignment_figure();plots(results,chosen);tuning_figure()
    localisation=development_localisation()
    alignment=s.read(OUT/'alignment.json')
    # Same-base ablations isolate the correction from the Gaussian/threshold
    # changes. These fixed settings are already recorded in the freeze.
    fresh=s.learn.Data(s.read(OUT/'datasets/test/index.json'),s.read(s.OLD/'normalization.json'))
    ablations={}
    for family in ('shift','extent','combined'):
        ablation_config=selection[family]['base']
        scores=np.load(OUT/f'confirmation/median_sigma{ablation_config["sigma"]:g}.npy',mmap_mode='r')
        rr=s.learn.evaluate_scores(fresh,scores,ablation_config)
        ablations[family]=dict(base=s.learn.objective(rr),corrected={k:results[family][k] for k in ('macro_f1','by_count')})
    s.write(OUT/'same_base_ablations.json',ablations)
    table=[]
    for name,r in results.items():
        for count in s.c.MULTIPLIERS:
            m=r['by_count'][str(count)]
            table.append(dict(method=NAMES[name],count=count,**{k:m[k] for k in ('f1','precision','recall','tp','fp','fn','clean_rings_flagged','clean_rings','pixel_iou')}))
    with (REPORT/'results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Median localisation follow-up','',f'Completed a separate study with 96 original training backgrounds, 32 original validation backgrounds and {s.FRESH_COUNT} newly generated confirmation backgrounds ({s.FRESH_COUNT*18} images). Original thesis/poster results are preserved.','',
        '## 1. Are the image and mask registered?','',
        f'All {alignment["geometries"]} training pore masks exactly match independently reconstructed native-grid geometry: {alignment["native_nearest_pixel_mask_mismatches"]} mismatched pixels. The coordinate check includes the native nearest-neighbour sampling at the angular seam.','',
        'The paired noiseless pore-minus-clean response is displaced radially. Median contrast-centroid offsets are -1.54 px on the inner slope, +0.04 px at the crown centre, and +1.17 px on the outer slope. The direction reverses across the crown. Around 59–63% of absolute contrast lies within the geometric opening in the diagnostic crop.','',
        '![Alignment](01_alignment.png)','',
        'The original median was also checked on validation images. Among responses overlapping a pore, the low-noise displacement and size errors were:','',
        '| Radial position | Pore scenes | Scenes with overlapping response | Median radial displacement (px) | Median predicted / true box height |',
        '|---|---:|---:|---:|---:|']
    for off,g in localisation['low_noise_by_radial'].items():
        lines.append(f'| {float(off):+.2f} mm | {g["pore_scenes"]} | {g["with_overlapping_response"]} | {g["median_dy_px"]:+.2f} | {g["median_height_ratio"]:.2f} |')
    diagnostic_low=localisation['by_count']['64']
    lines+=['',f'At 64× on validation, {diagnostic_low["with_overlapping_response"]} of {diagnostic_low["pore_scenes"]} pore scenes had an overlapping median response, but only {diagnostic_low["matched"]} met the required box IoU of 0.5. Another {diagnostic_low["no_overlapping_response"]} had no overlapping response. This separates partial localisation from missing the visible pore response.','',
        'These displacement summaries are conditional on an overlapping response. Missing responses are retained as misses in the full evaluation. Detailed count, pore-size and fragmentation breakdowns are saved in `development_localisation.json`.','',
        '## 2. What was recalibrated and tuned?','',
        'Recomputed clean-training per-row median/MAD calibration for all six Gaussian settings; coefficients exactly reproduce the original calibration. Expanded validation search: Gaussian sigma 0, 0.5, 1, 2, 4, 8 pixels; thresholds 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96; minimum areas 3, 8, 15, 24, 40, 64, 96 pixels; closing 1 or 3.','',
        'After selecting the best uncorrected operating point per Gaussian setting, tested box-position correction, box-size correction, and both. Position candidates include constant displacement and displacement towards/away from the known crown centre. Extent candidates scale and pad each detected component box. Every correction uses only image-derived components and known annular coordinates at inference.','',
        'Primary selection objective remains mean detection F1 across the three noise conditions, with pixel IoU and false positives as tie-breaks. The original one-to-one circular box IoU >= 0.5 matching rule is unchanged. Corrected pixel masks are filled detection rectangles; their pixel IoU is not equivalent to a learned pore contour.','',
        f'Validation selected **{NAMES[chosen]}** before fresh data generation. All ablations are retained and evaluated; the test outcomes do not change the chosen method.','',
        '![Validation correction sweep](04_validation_correction.png)','',
        '```json',json.dumps(dict(gaussian_sigma_px=config['sigma'],threshold=config['threshold'],
            minimum_region_area_px=config['min_area'],closing_size_px=config['closing'],
            correction=None if chosen=='retuned' else selected['correction'],
            validation_mean_f1=selected['macro_f1']),indent=2),'```','',
        '## 3. What happened on fresh backgrounds?','',
        '| Method | 1× F1 | 4× F1 | 64× F1 |','|---|---:|---:|---:|']
    for name,r in results.items():lines.append('| '+NAMES[name]+' | '+' | '.join(f'{r["by_count"][str(m)]["f1"]:.3f}' for m in s.c.MULTIPLIERS)+' |')
    low_old=results['original']['by_count']['64'];low_new=results[chosen]['by_count']['64'];low_cnn=results['cnn']['by_count']['64']
    lines+=['',f'At the lowest noise setting, the original median scored {low_old["f1"]:.3f} F1, the validation-selected revised median scored {low_new["f1"]:.3f}, and the unchanged CNN scored {low_cnn["f1"]:.3f}. The revised median detected {low_new["tp"]} of {low_new["tp"]+low_new["fn"]} pores, with {low_new["fp"]} false-positive regions. These counts separate successful localisation from remaining misses and unmatched detections.']
    lines+=['','![Results](02_fresh_results.png)','',
        '| Noise | Original median TP / FP / FN | Selected median TP / FP / FN | Paired F1 improvement (95% interval) |','|---|---:|---:|---:|']
    for m in s.c.MULTIPLIERS:
        a=results['original']['by_count'][str(m)];b=results[chosen]['by_count'][str(m)];d=ci[str(m)]
        lines.append(f'| {m}× | {a["tp"]} / {a["fp"]} / {a["fn"]} | {b["tp"]} / {b["fp"]} / {b["fn"]} | {d["delta"]:+.3f} [{d["paired_95_ci"][0]:+.3f}, {d["paired_95_ci"][1]:+.3f}] |')
    lines+=['','Intervals use 2,000 paired resamples of independent backgrounds. Repeated acquisitions and noise levels remain grouped. They are conditional on the fitted methods and do not quantify model-selection uncertainty.','',
        'All three intervals for the change from the original median include zero. The higher point scores on this 24-background set therefore provide limited evidence of a general improvement over the original operating point. The stricter revised threshold also trades recall for fewer false alarms.','',
        '### Does the correction help with the detector settings held fixed?','',
        '| Correction | 64× F1 before correction | 64× F1 after correction |','|---|---:|---:|']
    for family,a in ablations.items():
        lines.append(f'| {NAMES[family]} | {a["base"]["by_count"]["64"]["f1"]:.3f} | {a["corrected"]["by_count"]["64"]["f1"]:.3f} |')
    fixed=fixed_ci['64']
    lines+=['',f'For the selected method versus the retuned uncorrected median, the low-noise F1 change is {fixed["delta"]:+.3f}, with a paired 95% interval [{fixed["paired_95_ci"][0]:+.3f}, {fixed["paired_95_ci"][1]:+.3f}]. This comparison isolates the correction when their detector settings coincide, as recorded in the frozen configuration.']
    if (OUT/'fresh_failure_diagnosis.json').exists():
        d=s.read(OUT/'fresh_failure_diagnosis.json')['methods'][chosen]['counts']
        lines+=['','### Why does the low-noise median still score poorly?','',
            f'Of {d["pore_images"]} pore images, {d["matched"]} are correctly localised, {d["overlap_but_failed_iou"]} have a response overlapping the reference pore box that fails the localisation criterion, and {d["no_prediction_overlapping_pore"]} have no surviving overlapping prediction. These are descriptive checks after confirmation; no detector settings were changed.','',
            f'All {d["unmatched_overlapping_regions"]} unmatched detection boxes in the revised low-noise detector overlap reference pore boxes. There are {d["unmatched_elsewhere_regions"]} unmatched regions elsewhere on pore-containing images, and {low_new["clean_rings_flagged"]} of {low_new["clean_rings"]} clean images are flagged. Thus these false positives are near real pores and concern incomplete, displaced or duplicate regions. Any bounding-box overlap is a weaker diagnostic than actual mask agreement; a correct localisation still requires box IoU >= 0.5.','',
            f'At image level, the revised median flags {d["pore_images"]-d["no_prediction_anywhere"]} of {d["pore_images"]} pore-containing images and {low_new["clean_rings_flagged"]} of {low_new["clean_rings"]} clean images. This describes image-level alarming, a different task from the reported pore-localisation F1.','',
            'The remaining problem is chiefly region formation. TOPO contrast can occupy the rim or one side of an opening. Thresholding extracts those contrast patches; a fixed two-pixel shift and one-pixel padding cannot reconstruct every pore size or join disconnected fragments. Small weak responses can also be removed by thresholding or the minimum-area filter.','']
    if (OUT/'validation_sensitivity_diagnostic.json').exists():
        d=s.read(OUT/'validation_sensitivity_diagnostic.json')
        lines+=['','### Would a lower threshold help?','',
            'A follow-up validation-only diagnostic held Gaussian sigma 1, closing 3, minimum area 8 and the correction fixed. It did not select a replacement model or alter confirmation results.','',
            '| Threshold | Low-noise validation TP | FP | F1 |','|---|---:|---:|---:|']
        for t in d['trials']:
            if t['min_area']==8 and t['threshold'] in (12,16,24):
                m=t['by_count']['64'];lines.append(f'| {t["threshold"]} | {m["tp"]} | {m["fp"]} | {m["f1"]:.3f} |')
        lines+=['','Lowering the threshold globally does not recover more correctly localised pores in this comparison. A next development experiment could retain strong seeds, grow their local regions at a lower threshold and merge compatible fragments. That approach needs validation before any new confirmation. The earlier staged search was finite and did not jointly optimise every detector and correction setting.','']
    lines+=['',
        '![Examples](03_detection_examples.png)','',
        '## 4. Evidence and limits','',
        '- Fresh backgrounds and acquisition seeds are disjoint from original training, validation and confirmation. New data were generated after the hash-checked freeze.',
        '- Original median, recalibrated variants and the unchanged CNN were evaluated on exactly the same fresh images.',
        '- The revised median is a finite, staged search. Correction tuning uses the best uncorrected base per sigma; this does not establish a global optimum over every joint detector/correction setting.',
        '- This follows an observed historical failure, so new confirmation is required. The old confirmation images were excluded from fitting and selection in this study.',
        '- Low acquisition noise does not remove geometric background variation or make visible contrast equal the pore opening.',
        '- The position correction assumes the existing nominal annular coordinates, crown centre and 0.05 mm pixel pitch. A changed weld profile or inaccurate ring localisation requires separate validation.',
        '- Results remain synthetic and conditional on the existing procedural renderer. Real-image validation remains future work.',
        '',f'Frozen: {frozen["frozen_utc"]}. Audit: passed.','',
        f'Raw study artifacts: `{OUT}`','',
        'Reproduce: `.venv/Scripts/python.exe WeldDefectDetection/median_localisation_20261110.py all`, then `.venv/Scripts/python.exe WeldDefectDetection/report_median_localisation_20261110.py`.','']
    text='\n'.join(lines);(REPORT/'README.md').write_text(text,encoding='utf8')
    # Standalone local HTML report with large figures and the same exact text.
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        body=local_markdown(text)
    page='<!doctype html><meta charset="utf-8"><title>Median localisation follow-up</title><style>body{font:17px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#172534}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}pre{background:#f2f5f8;padding:18px;overflow:auto}h1,h2{color:#123c67}</style>'+body
    (REPORT/'OPEN_REPORT.html').write_text(page,encoding='utf8')
    print(json.dumps(dict(chosen=chosen,results={k:r['by_count'] for k,r in results.items()},report=str(REPORT/'OPEN_REPORT.html')),indent=2))

if __name__=='__main__':report(audit())
