"""Build human-readable results from the frozen four-approach campaign."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np

REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_comparison.campaign import DEFAULT, ARMS, OLD, read, require, check_protocol, verify, verify_files
from box_comparison.core import sha,write,summarize


def additional_uncertainty(root,primary):
    names=[primary,'all_background','fixed_central_band']
    rows={a:read(root/'predictions'/f'confirmation_{a}.json') for a in names}
    ids=sorted({r['sequence_id'] for r in rows[primary]})
    values={}
    for a,rr in rows.items():
        values[a]=np.array([[sum(r['pixel_tp'] for r in rr if r['sequence_id']==s),
                            sum(r['pixel_fp'] for r in rr if r['sequence_id']==s),
                            sum(r['pixel_fn'] for r in rr if r['sequence_id']==s),
                            sum(r['target_positive'] and r['predicted_positive'] for r in rr if r['sequence_id']==s),
                            sum(r['target_positive'] for r in rr if r['sequence_id']==s),
                            sum(not r['target_positive'] and not r['predicted_positive'] for r in rr if r['sequence_id']==s),
                            sum(not r['target_positive'] for r in rr if r['sequence_id']==s)] for s in ids])
    rng=np.random.default_rng(921990002);samples=[]
    for _ in range(2000):
        chosen=rng.integers(0,len(ids),len(ids));totals={a:v[chosen].sum(0) for a,v in values.items()}
        p=totals[primary]
        ba=(p[3]/max(p[4],1)+p[5]/max(p[6],1))/2
        iou={a:v[0]/max(sum(v[:3]),1) for a,v in totals.items()}
        samples.append([ba,iou[primary]-iou['all_background'],iou[primary]-iou['fixed_central_band']])
    return dict(interpretation='Supplementary descriptive sequence-cluster intervals; box F1 was the selection objective.',
        balanced_accuracy_interval=np.quantile(np.array(samples)[:,0],[.025,.975]).tolist(),
        pixel_iou_minus_background_interval=np.quantile(np.array(samples)[:,1],[.025,.975]).tolist(),
        pixel_iou_minus_band_interval=np.quantile(np.array(samples)[:,2],[.025,.975]).tolist())


def plots(root,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    vals=np.array([results['confirmation']['metrics'][a]['box_f1'] for a in ARMS])
    intervals=np.array([results['confirmation']['uncertainty']['intervals'][a]['box_f1'] for a in ARMS])
    ax.bar(ARMS,vals,color=['#777777','#2679b4','#37a482','#a65bb4'])
    # Draw percentile interval endpoints directly, since a point estimate need
    # not lie inside its percentile bootstrap interval in every possible run.
    ax.vlines(np.arange(4),intervals[:,0],intervals[:,1],color='black')
    ax.scatter(np.repeat(np.arange(4),2),intervals.ravel(),marker='_',color='black')
    ax.set_ylabel('Box F1 at IoU >= 0.5');ax.set_ylim(0,1)
    ax.set_title('Independent synthetic confirmation, 95% sequence-cluster intervals')
    fig.tight_layout();fig.savefig(root/'box_f1_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,arm in zip(axes,('cnn','cnn_median')):
        for path in sorted((root/'training').glob(f'{arm}_*.json')):
            r=read(path)
            if 'history' not in r:continue
            ax.plot([h['epoch'] for h in r['history']],[h['metrics']['box_f1'] for h in r['history']],label=str(r['seed']))
        ax.set_title(arm);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    axes[0].set_ylabel('Development validation box F1')
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    from box_comparison.core import load_windows
    from matplotlib.patches import Rectangle
    data=load_windows(root/'confirmation')
    index={m['window_id']:i for i,m in enumerate(data['meta'])}
    fig,axes=plt.subplots(4,2,figsize=(7,12));examples=[]
    for ri,arm in enumerate(ARMS):
        rr=read(root/'predictions'/f'confirmation_{arm}.json')
        for col,kind in enumerate(('fp','fn')):
            # Stable file order, no search for visually flattering examples.
            row=next((r for r in rr if r[kind]>0),None);ax=axes[ri,col]
            if row is None:ax.text(.1,.5,f'No {kind} cases');ax.axis('off');continue
            i=index[row['window_id']];ax.imshow(data['target'][i],cmap='gray',vmin=0,vmax=1)
            for key,color in [('truth_boxes','lime'),('boxes_xyxy','orange')]:
                for x0,y0,x1,y1 in row[key]:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=1))
            ax.set_title(f'{arm}: first {kind} case\n{row["window_id"]}',fontsize=8);ax.axis('off')
            examples.append(dict(approach=arm,error=kind,window_id=row['window_id'],tp=row['tp'],fp=row['fp'],fn=row['fn']))
    fig.tight_layout();fig.savefig(root/'error_examples.png',dpi=160);plt.close(fig)
    write(root/'error_examples.json',examples)


def main(root):
    root=Path(root).resolve();check_protocol(root);require(root,'evaluate');verify(root)
    results=read(root/'results.json');frozen=read(root/'frozen_models.json');visibility=read(root/'visibility_audit.json')
    additional=additional_uncertainty(root,results['primary']);write(root/'supplementary_intervals.json',additional)
    verify_files(read(OLD/'frozen_models.json')['checkpoint_hashes'])
    write(root/'historical_checkpoint_verification.json',dict(status='ok',files=read(OLD/'frozen_models.json')['checkpoint_hashes']))
    retries={}
    for name,folder in [('development',OLD/'datasets/sequence_dev'),('confirmation',root/'confirmation')]:
        sequences=read(folder/'sequence_manifest.json');groups={}
        for seq in sequences:
            label=seq['planned_class'];item=groups.setdefault(label,dict(sequences=0,rejected_attempts=0,total_attempts=0))
            item['sequences']+=1;item['rejected_attempts']+=seq['generation_attempt']-1;item['total_attempts']+=seq['generation_attempt']
        for item in groups.values():item['rejection_rate']=item['rejected_attempts']/item['total_attempts']
        retries[name]=groups
    write(root/'generation_retries.json',retries)
    records=[json.loads(line) for line in (root/'confirmation/samples.jsonl').read_text().splitlines()]
    domain=dict(energy_kev=sorted({r['realized_non_noise_parameters']['beam']['e0_kev'] for r in records}),
        accelerating_voltage_kv=sorted({r['realized_non_noise_parameters']['beam']['accelerating_voltage_kv'] for r in records}),
        material_names=sorted({r['realized_non_noise_parameters']['material']['name'] for r in records}),
        pixel_size_mm=sorted({r['pixel_size_mm'] for r in records}),
        emission_modes=sorted({r['emission_b_mode'] for r in records}),
        constant_b_values=sorted({r['emission_b_value'] for r in records}),
        layouts=sorted({r['renderer_parameters']['weld_layout']['mode'] for r in records}))
    if domain['energy_kev']!=[60] or domain['accelerating_voltage_kv']!=[60] or domain['pixel_size_mm']!=[.05]:
        raise ValueError('Actual confirmation operating settings differ from the intended domain.')
    write(root/'actual_operating_settings.json',domain)
    context_diagnostics={}
    for arm in ARMS:
        rr=read(root/'predictions'/f'confirmation_{arm}.json')
        context_diagnostics[arm]={}
        for key,condition in [('no_labelled_context_defect',lambda r:r['context_defective_count']==0),
                              ('at_least_one_labelled_context_defect',lambda r:r['context_defective_count']>0)]:
            context_diagnostics[arm][key]=summarize([r for r in rr if condition(r)])
    write(root/'context_contamination_diagnostics.json',context_diagnostics)
    import xml.etree.ElementTree as ET
    tests={}
    for filename in ('regression_tests.xml','inference_contract_tests.xml'):
        path=root/filename
        if path.exists():
            s=ET.parse(path).find('testsuite')
            tests[filename]={k:int(s.attrib[k]) for k in ('tests','errors','failures','skipped')}
            if tests[filename]['errors'] or tests[filename]['failures']:raise ValueError('Regression failures must be resolved before delivery.')
    write(root/'test_summary.json',tests)
    plots(root,results)
    metrics=results['confirmation']['metrics'];uncertainty=results['confirmation']['uncertainty']
    best=max(ARMS,key=lambda a:metrics[a]['box_f1'])
    rows=['# Four-approach TOPO bounding-box comparison','',
          f'The method selected before confirmation was **{results["primary"]}**. The highest observed confirmation box F1 was **{best}**, at **{metrics[best]["box_f1"]:.3f}**. These are different roles: the observed test ranking does not change the frozen primary method.','',
          'The experiment is complete as an operational synthetic comparison. These scores do not establish reliable industrial detection or transfer to measured SEM images. Constant B remains an explicit rendering assumption.','',
          '## Independent results','',
          'All four approaches used the same 350 target crops from 70 newly seeded sequences. Box matching required IoU >= 0.5. Intervals resample whole sequences.','',
          '| Approach | Box precision | Box recall | Box F1 [95% interval] | Pixel IoU | Clean targets with a box |',
          '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        m=metrics[a];lo,hi=uncertainty['intervals'][a]['box_f1']
        rows.append(f'| {a} | {m["box_precision"]:.3f} | {m["box_recall"]:.3f} | {m["box_f1"]:.3f} [{lo:.3f}, {hi:.3f}] | {m["pixel_iou"]:.3f} | {m["clean_false_alarms"]}/{m["clean_targets"]} |')
    rows+=['',
           f'At this matching threshold, the CNN correctly localised {metrics["cnn"]["tp"]} of {metrics["cnn"]["tp"]+metrics["cnn"]["fn"]} labelled regions and the hybrid {metrics["cnn_median"]["tp"]}. A predicted box only counts as correct when it meets the overlap requirement. Zero-valued bootstrap intervals arise when there are no observed matches; they do not establish that population performance is exactly zero.',
           '', '![Box F1 comparison](box_f1_comparison.png)','','## Does context help?','',
           '| Paired comparison | Box F1 difference, 95% interval | Interpretation |','|---|---:|---|']
    for name,values in uncertainty['paired_difference_intervals'].items():
        lo,hi=values['box_f1'];interpretation='Positive difference supported' if lo>0 else ('Negative difference supported' if hi<0 else 'Interval includes zero')
        rows.append(f'| {name} | [{lo:.3f}, {hi:.3f}] | {interpretation} |')
    rows+=['','These are exploratory, unadjusted pairwise intervals. Shuffled references come from another sequence without using labels. Thresholds and checkpoints remain unchanged. A hybrid-versus-CNN difference and a correct-versus-shuffled-context difference answer related but distinct questions.',
           '', ('The hybrid demonstrates a positive box-F1 improvement over the target-only CNN in this unadjusted comparison.'
                if uncertainty['paired_difference_intervals']['cnn_median_minus_cnn']['box_f1'][0]>0 else
                'The hybrid does not demonstrate a positive box-F1 improvement over the target-only CNN. Compare false alarms and region recall together before interpreting greater selectivity as better detection.'), '',
           '## The three repairs','',
           '1. **Feature and filter assumptions:** thresholding now permits bright and dark morphological responses, uses the whole crop, and has no maximum-area or aspect-ratio rejection. Validation selects the kernel, response polarity, threshold, minimum area and optional opening.',
           '2. **Consistent selection:** every approach uses validation box F1, with pixel IoU and fewer false positives as tie breakers. CNN checkpoint selection uses that same objective. No confirmation labels are used to select parameters.',
           '3. **Visibility audit:** weak-contrast cases remain in the data. The historical policy was diagnosed on the shared crop population, and six noiseless defect-on/off replays reproduced saved masks exactly. A nonzero causal renderer response is not a guarantee that every noisy target is detectable.','',
           '| Source class | Weak contrast proxy / labelled targets | Mean fraction of truth retained by old bright ROI | Truth regions rejected by old size/shape rules |',
           '|---|---:|---:|---:|']
    for c,r in visibility['classes'].items():
        rows.append(f'| {c} | {r["weak_contrast_proxy_count"]}/{r["targets"]} | {r["roi_truth_fraction"]:.3f} | {r["old_size_shape_rejected_regions"]}/{r["true_regions"]} |')
    rows+=['','The weak proxy means absolute median mask-to-annulus contrast divided by robust annulus variation is below one. It is a diagnostic, not a label-removal rule. Historical morphology measurements above concern these 64px development crops and should not be substituted for the old 128px campaign results.','',
           '## Per-class confirmation box F1','',
           '| Sequence source class | Thresholding | CNN | Median | CNN + median |','|---|---:|---:|---:|---:|']
    for c in sorted(results['confirmation']['per_class']['cnn']):
        if c=='clean':continue
        rows.append('| '+c+' | '+' | '.join(f'{results["confirmation"]["per_class"][a][c]["box_f1"]:.3f}' for a in ARMS)+' |')
    rows+=['','Class groups use the sequence source class. Some sequences contain a second injected defect. Clean-target false alarms above include every target without a labelled region, including clean crops from defective sequences.','',
           '## Frozen model settings','']
    for a in ARMS:
        s=frozen['models'][a]
        rows.append(f'- **{a}:** `{json.dumps(s["parameters"],sort_keys=True)}`'+(f'; seed {s["seed"]}, epoch {s["epoch"]}.' if 'seed' in s else '.'))
    rows+=['','There were three seeds and ten epochs per CNN family. The network uses detailed masks for supervised learning, then exports connected-region bounding boxes. The old learned context reconstruction model is preserved but is not an arm in this comparison.','',
           '![Training curves](training_curves.png)','','## Examples','',
           'Each row below uses the largest labelled target in its source class. Selection uses masks, not successful predictions. Green boxes are ground truth; orange boxes are predictions.','',
           '![Confirmation examples](confirmation_examples.png)','',
           'The following panel shows the first false-positive and missed-region case in saved target order for each method. Both kinds of error may occur in one image.','',
           '![Error examples](error_examples.png)','','## Evidence and reproducibility','',
           '- `protocol.json` records the population, selection policy, parameter grids, seeds, source hashes and assumptions.',
           '- `source_snapshot.zip` archives the source used for the campaign.',
           '- `development_audit.json`, `confirmation_audit.json` and the inventories record dataset validation and artifact hashes.',
           '- `frozen_models.json` binds normalization, model parameters and checkpoint hashes before confirmation.',
           '- `results.json` contains full metrics, class results, bootstrap intervals and baseline controls.',
           '- `predictions/` contains boxes, ground-truth regions and evaluation counts for every target.',
           '- `counterfactual/` contains the matched defect-on/off audit examples.',
           '- `inference_smoke/input.npz` contains raw inputs and geometry metadata without labels. All four inference outputs reproduce evaluation boxes.',
           '- `verification.json` confirms artifact integrity and preserved historical files. The two historical selected checkpoint hashes also match their original frozen record.',
           f'- Regression and inference-contract tests: **{sum(t["tests"] for t in tests.values())} passed** across the XML test reports. `test_summary.json` records their counts.',
           '- `generation_retries.json` reports per-class rejected attempts, including difficult cases resampled by the existing generator.',
           '- `context_contamination_diagnostics.json` separates targets with and without labelled defects in their neighbours. Those labels are used only for this audit, never for reference construction.',
           '- `WeldDefectDetection/BOX_CONTEXT_COMPARISON.md` explains the formulas, annotation policy and commands.','',
           '## Remaining scope','',
           'This delivery completes the requested four-way synthetic comparison. The observed errors and uncertainty determine how strongly its results can be used in the thesis. More training after viewing these confirmation results would require a new development decision and a new confirmation seed. Circular-weld registration, measured SEM validation and physical qualification of B remain separate work.','']
    (root/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    files=[root/'REPORT.md',root/'results.json',root/'supplementary_intervals.json',root/'verification.json',
           root/'protocol.json',root/'frozen_models.json',root/'source_snapshot.zip',root/'regression_tests.xml',root/'inference_contract_tests.xml',root/'resumption_check.log',
           root/'test_summary.json',root/'generation_retries.json',root/'historical_checkpoint_verification.json',root/'context_contamination_diagnostics.json',root/'actual_operating_settings.json',
           root/'box_f1_comparison.png',root/'training_curves.png',root/'confirmation_examples.png',root/'error_examples.png',root/'error_examples.json',
           REPO/'WeldDefectDetection/BOX_CONTEXT_COMPARISON.md',Path(__file__),
           REPO/'WeldDefectDetection/tests/test_box_comparison.py',REPO/'WeldDefectDetection/tests/test_box_inference_contract.py',
           *(root/'receipts').glob('*.json'),*(root/'inference_smoke').glob('*')]
    write(root/'FINAL_MANIFEST.json',dict(schema='box-comparison-delivery/v1',files={str(p.resolve()):sha(p) for p in files if p.is_file()},
        dataset_inventories={str(root/'development_inventory.json'):sha(root/'development_inventory.json'),
                             str(root/'confirmation_inventory.json'):sha(root/'confirmation_inventory.json')},
        physical_qualification=False))
    print(str(root/'REPORT.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    main(parser.parse_args().root)
