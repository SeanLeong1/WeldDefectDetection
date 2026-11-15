"""Generate reviewable circular-CV results and actual inference-stage figures."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import run_circular_campaign_20261004 as run
from box_development.methods import aligned_reference, residual, make_network, scores_for
from box_refinement.methods import predicted_regions
from box_comparison.core import write, sha

OUT=run.OUT
FIG=run.ROOT/'docs/figures/circular_campaign_20261004'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def layer_audit(spec, target):
    model=make_network(1)
    model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)['state_dict'])
    model.eval();rows=[];features={}
    handles=[]
    for name,module in model.named_modules():
        if not list(module.children()):
            def hook(mod,args,out,name=name):
                rows.append(dict(name=name,type=type(mod).__name__,input=list(args[0].shape),output=list(out.shape),
                                 parameters=sum(p.numel() for p in mod.parameters(recurse=False))))
            handles.append(module.register_forward_hook(hook))
        if name in ('enc1','enc2','bottleneck','dec2','dec1','head'):
            def capture(mod,args,out,name=name):features[name]=out.detach().numpy()[0]
            handles.append(module.register_forward_hook(capture))
    with torch.inference_mode():model(torch.from_numpy(target[None,None]))
    for h in handles:h.remove()
    write(OUT/'cnn_layer_audit.json',dict(parameters=sum(p.numel() for p in model.parameters()),layers=rows,
        checkpoint_sha256=spec['checkpoint_sha256'],feature_plot='Mean absolute channel activation, independently scaled per panel; not a physical interpretation.'))
    fig,axes=plt.subplots(1,6,figsize=(15,3),layout='constrained')
    for ax,(name,value) in zip(axes,features.items()):
        shown=np.mean(np.abs(value),axis=0) if name!='head' else 1/(1+np.exp(-value[0]))
        ax.imshow(shown,cmap='viridis');ax.set_title(f'{name}\n{value.shape[0]} x {value.shape[1]} x {value.shape[2]}');ax.axis('off')
    fig.suptitle('Actual selected CNN activations: mean absolute features; panels use independent scales')
    fig.savefig(FIG/'cnn_features.png',dpi=170);plt.close(fig)


def rings(norm):
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    for ax,kind in zip(axes,run.c.KINDS):
        with np.load(OUT/f'datasets/test/test-000-{kind}.ring.npz') as a:
            raw=a['raw'];mask=a['mask']
        image=np.sqrt(np.clip((raw-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1))
        ax.set_facecolor('#e5e7eb')
        ax.imshow(image,cmap='gray',vmin=0,vmax=1,extent=[-48,48,-48,48],origin='lower')
        if mask.any():
            yy,xx=np.where(mask)
            ax.plot((xx.mean()-960)*.05,(yy.mean()-960)*.05,'o',mfc='none',mec='#d55e00',ms=13,mew=1.5)
        ax.set_title(kind.replace('_',' '));ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    fig.suptitle('Held-out test ring 000: 80 mm diameter; grey exterior is outside the inspected band')
    fig.savefig(FIG/'circular_dataset.png',dpi=180);plt.close(fig)


def result_plot(results):
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    x=np.arange(2);width=.34
    for ax,key,label in zip(axes,('box_f1','box_recall'),('Box F1','Box recall')):
        old=[results['straight_trained_same_test'][m][key] for m in ('cnn','median')]
        new=[results['noisy']['metrics'][m][key] for m in ('cnn','median')]
        ci=[results['noisy']['uncertainty']['intervals'][m][key] for m in ('cnn','median')]
        ax.bar(x-width/2,old,width,label='Straight-trained transfer',color='#7c8797')
        ax.bar(x+width/2,new,width,label='Circular-trained',color='#0072b2')
        # Percentile cluster intervals may not be symmetric about the observed value.
        for pos,(low,high) in zip(x+width/2,ci):
            ax.plot([pos,pos],[low,high],color='black',lw=1)
            ax.plot([pos-.04,pos+.04],[low,low],color='black',lw=1)
            ax.plot([pos-.04,pos+.04],[high,high],color='black',lw=1)
        ax.set_xticks(x,['CNN','Neighbour median']);ax.set_ylim(0,1.08);ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Same held-out circular test set; circular-model whiskers: 95% ring-cluster intervals')
    fig.savefig(FIG/'comparison.png',dpi=180);plt.close(fig)


def examples(models):
    data=run.load('test');scores={m:scores_for(data,s)[:,0] for m,s in models.items()}
    ids=[next(i for i,m in enumerate(data['meta']) if m['source_class']==kind and data['mask'][i].any()) for kind in run.c.KINDS[1:]]
    clean=[i for i,m in enumerate(data['meta']) if m['source_class']=='clean']
    # Deliberately show a difficult clean example rather than selecting a flattering one.
    ids.append(max(clean,key=lambda i:float(scores['cnn'][i].max())))
    fig,axes=plt.subplots(3,5,figsize=(13,8),layout='constrained')
    selection=[]
    for rr,i in enumerate(ids):
        t=data['target'][i];ref=aligned_reference(t,data['context'][i]);truth=data['mask'][i]
        panels=[(t,'Input + truth boundary','gray',0,1),(ref,'Aligned neighbour median','gray',0,1),
                (scores['median'][i],'Median anomaly score','magma',0,8),
                (scores['cnn'][i],'CNN sigmoid score','viridis',0,1),(t,'Predicted boxes','gray',0,1)]
        for ax,(im,title,cmap,low,high) in zip(axes[rr],panels):
            ax.imshow(im,cmap=cmap,vmin=low,vmax=high)
            if rr==0:ax.set_title(title)
            ax.set_xticks([]);ax.set_yticks([])
        if truth.any():axes[rr,0].contour(truth,levels=[.5],colors=['#d55e00'],linewidths=1)
        axes[rr,0].set_ylabel(data['meta'][i]['source_class'].replace('_',' '))
        for method,color in (('cnn','#00a6ff'),('median','#ff9f1c')):
            _,boxes=predicted_regions(scores[method][i],**models[method]['parameters'])
            for x0,y0,x1,y1 in boxes:axes[rr,4].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=color,lw=1.7))
        selection.append(data['meta'][i])
    fig.suptitle('Actual held-out inference: cyan = CNN boxes; orange = median boxes\nFirst positive sector of each pore type; clean sector with largest CNN peak')
    fig.savefig(FIG/'method_steps.png',dpi=180);plt.close(fig)
    write(OUT/'illustrated_windows.json',selection)
    layer_audit(models['cnn'],data['target'][ids[0]])


def main():
    torch.set_num_threads(4);FIG.mkdir(parents=True,exist_ok=True)
    results=read(OUT/'results.json');audit=read(OUT/'dataset_audit.json');norm=read(OUT/'normalization.json')
    models=read(OUT/'frozen_models.json')['models']
    ring_stats={}
    for domain,prefix in (('Circular','noisy'),('Straight transfer','straight_transfer')):
        ring_stats[domain]={}
        for method in ('cnn','median'):
            rows=[r for r in read(OUT/f'predictions/{prefix}_{method}.json') if r['source_class']=='clean']
            ids=sorted({r['sequence_id'] for r in rows})
            flagged=sum(any(r['predicted_positive'] for r in rows if r['sequence_id']==sid) for sid in ids)
            ring_stats[domain][method]=dict(clean_rings=len(ids),flagged_clean_rings=flagged,
                clean_ring_false_alarm_rate=flagged/len(ids),false_boxes_per_clean_ring=sum(r['fp'] for r in rows)/len(ids))
    write(OUT/'ring_level_summary.json',ring_stats)
    rings(norm);result_plot(results);examples(models)
    layers=read(OUT/'cnn_layer_audit.json');population=read(OUT/'training_population.json')
    lines=['# Circular-only weld dataset and CV results','',
        'Completed 4 October 2026. Synthetic annular inspection of 80 mm diameter welds, 0.05 mm Cartesian sampling.','',
        '## Previous training','',
        'The previous circular examples used frozen straight-weld models without circular tuning. This experiment fits the CNN and median detector on new circular data and compares both with the old models on the same held-out rings.','',
        '## Dataset','',
        '| Split | Independent backgrounds | Scenes | Sector crops |','|---|---:|---:|---:|']
    for s in ('train','val','test'):
        lines.append(f'| {s} | {audit["split_backgrounds"][s]} | {audit["split_scenes"][s]} | {audit["split_sectors"][s]} |')
    lines+=['',f'Total: {audit["backgrounds"]} backgrounds, {audit["scenes"]} scenes, {audit["sectors"]} sectors. Each background has clean/keyhole/intrinsic variants. Variation levels are 0%, 5% and 10%. All related data stay in one split.',
        '',f'CNN optimization used {population["cnn_selected_sectors"]} training sectors, including {population["cnn_positive_sectors"]} positive sectors and eight sampled negative sectors per scene. Median calibration uses clean training windows. Validation and testing use all sectors.',
        '',f'Three CNN seeds trained for 20 epochs each. Selected seed {models["cnn"]["seed"]}, epoch {models["cnn"]["epoch"]}. Network parameters: {layers["parameters"]:,}.','',
        '![Dataset](figures/circular_campaign_20261004/circular_dataset.png)','',
        '## Noisy held-out results','',
        '| Training domain | Method | TP / FP / FN | Box precision | Box recall | Box F1 | Pixel IoU | Clean-source sector FA |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for domain,metrics in [('Straight transfer',results['straight_trained_same_test']),('Circular',results['noisy']['metrics'])]:
        for m in ('cnn','median'):
            s=metrics[m]
            lines.append(f'| {domain} | {m} | {s["tp"]} / {s["fp"]} / {s["fn"]} | {s["box_precision"]:.3f} | {s["box_recall"]:.3f} | {s["box_f1"]:.3f} | {s["pixel_iou"]:.3f} | {100*s["clean_source_false_alarm_rate"]:.2f}% |')
    for m in ('cnn','median'):
        new=results['noisy']['metrics'][m]['box_f1'];old=results['straight_trained_same_test'][m]['box_f1']
        relation='higher' if new>old else 'lower' if new<old else 'equal'
        lines+=['',f'The circular-trained {m} has {relation} observed box F1 than its straight-trained counterpart on these held-out rings ({new:.3f} versus {old:.3f}).']
    lines+=['','Box matching uses one-to-one IoU >= 0.5. Counts refer to connected sector regions, not necessarily physical pores. A pore crossing a sector boundary can contribute more than one truth region.',
        '', 'The same-test comparison measures these saved models. Their training populations and budgets differ, so any difference cannot be attributed solely to circular versus straight training geometry.','',
        '![Same-test comparison](figures/circular_campaign_20261004/comparison.png)','',
        '| Circular-trained method | Box F1 95% cluster interval | Recall 95% cluster interval |','|---|---:|---:|']
    for m in ('cnn','median'):
        ci=results['noisy']['uncertainty']['intervals'][m]
        lines.append(f'| {m} | {ci["box_f1"][0]:.3f} to {ci["box_f1"][1]:.3f} | {ci["box_recall"][0]:.3f} to {ci["box_recall"][1]:.3f} |')
    lines+=['','| Training domain | Method | Clean rings flagged at least once | False boxes per clean ring |','|---|---|---:|---:|']
    for domain,methods in ring_stats.items():
        for m,s in methods.items():
            lines.append(f'| {domain} | {m} | {s["flagged_clean_rings"]}/{s["clean_rings"]} | {s["false_boxes_per_clean_ring"]:.2f} |')
    if (OUT/'test_background_diagnostics.json').exists():
        diagnostics=read(OUT/'test_background_diagnostics.json');worst=diagnostics[0]
        sid=worst['background'];quiet_rows=read(OUT/'predictions/quiet_median.json')
        quiet_fp=sum(r['fp'] for r in quiet_rows if r['sequence_id']==sid)
        lines+=['',f'Failure concentration: {sid} contributes {worst["metrics"]["median"]["fp"]} of {results["noisy"]["metrics"]["median"]["fp"]} median false boxes across its three scenes. It still contributes {quiet_fp} false boxes with acquisition noise removed. This implicates a mismatch between normal background structure and the calibrated reference/scale, rather than added noise alone. Separating texture, profile and registration effects requires a controlled development experiment. No test-based retuning was performed.']
    lines+=['','Intervals use 2,000 whole-background bootstrap draws over only 12 test backgrounds. They do not include retraining variability or real-image domain shift.','',
        '| Pore source | Method | TP / FP / FN | Box F1 | Recall |','|---|---|---:|---:|---:|']
    for kind in run.c.KINDS[1:]:
        for m in ('cnn','median'):
            s=results['noisy']['per_pore'][m][kind]
            lines.append(f'| {kind} | {m} | {s["tp"]} / {s["fp"]} / {s["fn"]} | {s["box_f1"]:.3f} | {s["box_recall"]:.3f} |')
    lines+=['','Per-pore rows include every sector in scenes of that pore type, including empty sectors, so false positives away from the pore are retained.','',
        '## Frozen detection settings','',
        '| Method | Score threshold | Minimum area (pixels) | Closing kernel |','|---|---:|---:|---:|']
    for m in ('cnn','median'):
        p=models[m]['parameters']
        lines.append(f'| {m} | {p["threshold"]} | {p["min_area"]} | {p["closing"]} x {p["closing"]} |')
    lines+=['','| CNN initialization seed | Selected epoch | Validation box F1 |','|---|---:|---:|']
    for candidate in read(OUT/'model_candidates.json')['cnn']:
        lines.append(f'| {candidate["seed"]} | {candidate["epoch"]} | {candidate["metrics"]["box_f1"]:.3f} |')
    lines+=['','## Learn the methods','',
        'Read [the chronological method guide](CIRCULAR_CV_METHODS_20261004.md) for formulas, every CNN stage, training and interpretation.','',
        '![Inference steps](figures/circular_campaign_20261004/method_steps.png)','',
        '![Actual CNN features](figures/circular_campaign_20261004/cnn_features.png)','',
        'Feature panels show mean absolute channel activations with independent display ranges; brightness across panels is not comparable and no individual feature is assigned a physical meaning.','',
        '## Verification and limits','',
        '- Full-frame versus padded-annulus pilot: zero noiseless crop error and identical labels for clean and both pore types.',
        '- 29 numerical/detection tests passed, including exact quadrature reuse, ring coordinate sampling, matching and inference safeguards.',
        '- Dataset audit: disjoint background seeds, all circular layouts, raster_v2 acquisition, and all scene hashes verified.',
        '- Model and preprocessing hashes frozen before test generation. Test data never select checkpoints or thresholds.',
        '- Only the annular inspection band is valid. Grey exterior in ring figures is unmeasured; small sector gaps remain.',
        '- Synthetic evidence only. Assumed emission/acquisition parameters, no real SEM calibration, internal-pore or acceptance validation.','',
        '## Reproduce and inspect','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe WeldDefectDetection\\run_circular_campaign_20261004.py --stage all',
        '.\\.venv\\Scripts\\python.exe WeldDefectDetection\\report_circular_campaign_20261004.py','```','',
        f'Campaign directory: `{OUT}`. `results.json` includes quiet/noisy metrics, amplitudes and uncertainty. `predictions/` contains every decision. `datasets/` contains float32 raw/quiet sectors, masks and metadata. `cnn_layer_audit.json` lists runtime shapes and parameter counts for every leaf layer.','']
    report=run.ROOT/'docs/CIRCULAR_CV_RESULTS_20261004.md';report.write_text('\n'.join(lines),encoding='utf-8')
    dataset_rows=[r for s in run.COUNTS for r in read(OUT/f'datasets/{s}/index.json')]
    write(OUT/'dataset_manifest.json',dict(schema='circular-sector-dataset/v1',
        arrays=dict(raw='float32 [78,64,64], noisy TOPO',quiet='float32 [78,64,64], pre-noise TOPO',mask='uint8 [78,64,64], geometric defect labels'),
        records=[dict(scene_id=r['scene_id'],background_id=r['background_id'],split=r['split'],seed=r['seed'],
            source_class=r['source_class'],amplitude=r['amplitude'],path=f'datasets/{r["split"]}/{r["scene_id"]}.npz',sha256=r['npz_sha256']) for r in dataset_rows]))
    (OUT/'README.md').write_text('\n'.join([
        '# Circular-only weld dataset','',
        '45 independent synthetic circular backgrounds; 135 clean/keyhole/intrinsic scenes; 10,530 sector crops.',
        'All geometry is an 80 mm diameter ring at 0.05 mm Cartesian sampling. The valid inspection region is the annular band, not the whole plate.','',
        'Use dataset_manifest.json for portable relative paths, splits, seeds, classes and SHA-256 values.',
        'Each scene NPZ contains raw, quiet and mask arrays, all shaped [78, 64, 64]. Raw/quiet are float32 TOPO; masks are uint8.',
        'Each sector uses its four distinct neighbours at offsets -2,-1,+1,+2 modulo 78 for median inference.',
        'Keep all scenes/sectors sharing a background_id in one split. Use the saved training normalization for every split.','',
        '```python','import json','import numpy as np','from pathlib import Path',
        "root = Path(__file__).resolve().parent  # when this example is saved in this directory",
        "manifest = json.loads((root / 'dataset_manifest.json').read_text())",
        "record = manifest['records'][0]",
        "with np.load(root / record['path']) as scene:",
        "    raw = scene['raw']   # 78 sectors, unnormalized float32",
        "    truth = scene['mask']",
        "norm = json.loads((root / 'normalization.json').read_text())",
        "images = np.sqrt(np.clip((raw - norm['raw_low']) / (norm['raw_high'] - norm['raw_low']), 0, 1))",
        'target = images[0]','context = images[[76, 77, 1, 2]]','```','',
        'results.json: noisy, quiet, same-test straight-trained transfer, pore types, amplitudes and uncertainty.',
        'frozen_models.json: chosen settings, checkpoint paths and hashes. training/: all three CNN checkpoints.',
        'predictions/: per-sector truth boxes, predicted boxes, counts and metadata.',
        'protocol.json and source_snapshot.zip: fixed experiment and implementation.',
        'pilot.json and dataset_audit.json: numerical and integrity checks.',
        'cnn_layer_audit.json: every executed leaf layer, runtime tensor dimensions and parameter counts.','',
        'Synthetic benchmark only. No measured SEM calibration or manufacturing acceptance validation.','']),encoding='utf-8')
    write(OUT/'delivery_manifest.json',dict(report=str(report),guide=str(run.ROOT/'docs/CIRCULAR_CV_METHODS_20261004.md'),
        files={str(p):sha(p) for p in [report,Path(__file__),run.ROOT/'docs/CIRCULAR_CV_METHODS_20261004.md',
            OUT/'dataset_manifest.json',OUT/'README.md',OUT/'ring_level_summary.json',OUT/'cnn_layer_audit.json',*FIG.glob('*.png')]}))
    print(report)


if __name__=='__main__':main()
