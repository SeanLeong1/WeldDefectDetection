"""Report measured outputs from the frozen Gaussian/consensus experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import run_gaussian_consensus_20261020 as e

FIG = e.ROOT/'docs/figures/gaussian_consensus_20261020'
DOC = e.ROOT/'docs/GAUSSIAN_CONSENSUS_RESULTS_20261020.md'
LABELS = {'median':'Continuous median','consensus':'Eight-hypothesis consensus',
          'cnn_same_scale':'CNN trained at selected scale',
          'original_median_unfiltered':'Retained original median',
          'original_cnn_unfiltered':'Retained original CNN'}


def verify_results(selection, result):
    """Check paired populations and reconstruct saved example detections."""
    index=e.read(e.OUT/'confirmation/index.json')
    expected={m['scene_id'] for m in index}
    assert len(expected)==96
    for b in {m['background_id'] for m in index}:
        group=[m for m in index if m['background_id']==b]
        assert {m['source_class'] for m in group}==set(e.previous.KINDS)
        assert len({(m['seed'],m['dwell_bin'],m['dwell_time_us'],m['amplitude']) for m in group})==1
    confirmation_seeds={m['seed'] for m in index}
    for source in (e.original.OUT,e.previous.OUT,e.prior.OUT,e.base.c.OUT):
        for p in list(source.glob('datasets/*/index.json'))+list(source.glob('confirmation/index.json')):
            assert not confirmation_seeds.intersection(m['seed'] for m in e.read(p))
    sigma=max(e.SIGMAS,key=lambda s:(*e.metrics.key(selection['trials'][str(s)]['median']['primary_pores']),-s))
    assert sigma==selection['selected_sigma']==result['selected_sigma']
    for name,r in result['results'].items():
        assert {row['scene_id'] for row in r['rows']}==expected
        assert e.metrics.summary(r['rows'])==r['all_conditions']
        for row in r['rows']:
            counted=e.metrics.match(row['predicted_boxes'],row['truth_boxes'],5024)
            assert all(row[k]==counted[k] for k in ('tp','fp','fn'))
    reconstructed=0
    for meta in index:
        if meta['index']!=0:continue
        with np.load(meta['npz_path']) as z:truth=z['mask'][16:144]
        with np.load(e.OUT/'predictions'/(meta['scene_id']+'_scores.npz')) as z:
            for name in LABELS:
                score=z[name]
                assert score.shape==truth.shape and np.isfinite(score).all() and score.min()>=0
                if 'cnn' in name:assert score.max()<=1
                op=e.CNN_OP if 'cnn' in name else e.OP
                row=e.prior.rows_for(score,truth,meta,op,True)['0.50']
                saved=next(r for r in result['results'][name]['rows'] if r['scene_id']==meta['scene_id'])
                assert row==saved
                reconstructed+=1
    return dict(status='passed',paired_backgrounds=24,identical_scene_sets=True,
                seed_overlap=False,selection_recomputed=True,box_match_rows_recomputed=480,
                example_score_reconstructions=reconstructed,example_scores_finite=True)


def figures(selection, result):
    FIG.mkdir(parents=True,exist_ok=True)
    mm = np.array(e.SIGMAS)*.05
    fig,axes = plt.subplots(1,2,figsize=(12,4.6))
    for name,col in [('median','#c97416'),('consensus','#7550ae')]:
        records=[selection['trials'][str(s)][name]['primary_pores'] for s in e.SIGMAS]
        axes[0].plot(mm,[r['f1'] for r in records],'o-',color=col,label=LABELS[name])
        axes[1].plot(mm,[r['clean_rings_flagged'] for r in records],'o-',color=col)
    for ax in axes:
        ax.axvline(selection['selected_sigma']*.05,color='#444',ls=':',label='Median-selected scale')
        ax.set_xlabel('Gaussian standard deviation, sigma (mm at ring centreline)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation pore-task box F1 (IoU >= 0.5)')
    axes[1].set_ylabel('Clean validation rings flagged, out of 24')
    axes[0].legend(fontsize=8)
    fig.suptitle('Finite scale selection: six candidates, unchanged detection rules')
    fig.tight_layout()
    fig.savefig(FIG/'validation_scale.png',dpi=200);fig.savefig(FIG/'validation_scale.pdf');plt.close(fig)

    names=['original_median_unfiltered','median','consensus','cnn_same_scale','original_cnn_unfiltered']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    colors=['#999','#c97416','#7550ae','#008d9b','#555']
    x=np.arange(len(names))
    for ax,pop,title in zip(axes,['primary_pores','all_conditions'],
                           ['Pore task: 24 clean + 48 pore scenes','All conditions: also includes 24 bulges']):
        values=[result['results'][n][pop]['f1'] for n in names]
        bars=ax.bar(x,values,color=colors)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3)
        ax.set_xticks(x,['Original\nmedian','Smoothed\nmedian','Smoothed\nconsensus','Same-scale\nCNN','Original\nCNN'],fontsize=8)
        ax.set_ylim(0,1);ax.set_ylabel('Fresh confirmation box F1 (IoU >= 0.5)')
        ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'Fresh 24-background comparison | selected sigma = {result["selected_sigma"]*.05:g} mm')
    fig.tight_layout();fig.savefig(FIG/'confirmation.png',dpi=200);fig.savefig(FIG/'confirmation.pdf');plt.close(fig)

    index=e.read(e.OUT/'confirmation/index.json')
    pair=[next(m for m in index if m['index']==0 and m['source_class']==kind)
          for kind in ('clean','open_intrinsic')]
    with np.load(pair[1]['npz_path']) as z:
        yy,xx=np.where(z['mask'][16:144])
    # Circular mean avoids a discontinuous ROI for a pore crossing the seam.
    centre=int(np.round((np.angle(np.exp(2j*np.pi*xx/5024).mean())%(2*np.pi))*5024/(2*np.pi)))%5024
    cols=(centre+np.arange(-64,64))%5024
    fig,axes=plt.subplots(2,6,figsize=(16.5,7))
    for i,m in enumerate(pair):
        with np.load(m['npz_path']) as z:raw=z['raw'][0];truth=z['mask'][16:144][:,cols]
        original=e.preprocess(raw,0)[16:144][:,cols]
        smooth=e.preprocess(raw,result['selected_sigma'])[16:144][:,cols]
        with np.load(e.OUT/'predictions'/(m['scene_id']+'_scores.npz')) as z:
            masks={n:e.metrics.binary(z[n],**{'threshold':e.CNN_OP['threshold'] if n=='cnn_same_scale' else e.OP['threshold'],
                 'area':15,'closing':3})[0][:,cols] for n in ('median','consensus','cnn_same_scale')}
        for j,ax in enumerate(axes[i]):
            ax.imshow(original if j==0 else truth if j==2 else smooth,cmap='gray',vmin=0,vmax=1,origin='lower',
                      extent=(-3.2,3.2,-3.2,3.2))
            ax.set_xticks([-3,0,3]);ax.set_yticks([-3,0,3]);ax.tick_params(labelsize=8)
            if j>=3:
                name=('median','consensus','cnn_same_scale')[j-3];mask=masks[name]
                overlay=np.zeros((*mask.shape,4));rgb=matplotlib.colors.to_rgb(colors[j-2])
                overlay[mask>0]=(*rgb,.75)
                ax.imshow(overlay,origin='lower',extent=(-3.2,3.2,-3.2,3.2))
                if truth.any():ax.contour(truth,levels=[.5],colors=['#38e668'],linewidths=1,
                                         origin='lower',extent=(-3.2,3.2,-3.2,3.2))
                r=next(r for r in result['results'][name]['rows'] if r['scene_id']==m['scene_id'])
                ax.set_xlabel(f'Full ring: TP {r["tp"]}, FP {r["fp"]}, FN {r["fn"]}',fontsize=8)
        axes[i,0].set_ylabel(('Clean' if i==0 else 'Injected pore')+'\nRadial offset (mm)')
    for ax,title in zip(axes[0],['Original TOPO','Gaussian TOPO','Geometry mask','Median detection','Consensus detection','CNN detection']):
        ax.set_title(title,fontsize=11)
    fig.suptitle('Prespecified first fresh background: matched clean and pore views',fontsize=15)
    fig.text(.5,.065,'Local images share intensity scale. Green outline = true pore; colour fill = predicted defect.\n'
             'Coordinates in mm; counts refer to the complete scored annulus. This example was not chosen for its outcome.',
             ha='center',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.18,wspace=.2,hspace=.35)
    fig.savefig(FIG/'matched_example.png',dpi=220);fig.savefig(FIG/'matched_example.pdf');plt.close(fig)


def main():
    e.verify()
    assert e.read(e.OUT/'audit.json')['status']=='passed'
    selection=e.read(e.OUT/'selection.json');result=e.read(e.OUT/'results.json')
    verification=verify_results(selection,result)
    figures(selection,result)
    sigma=result['selected_sigma']
    text=[
        '# Gaussian scale, median, consensus and CNN experiment',
        '',
        f'Campaign: `{e.OUT.name}`. Synthetic raised circular welds, declared nominal coordinates. '
        '24 independent fresh backgrounds, four conditions each (96 scenes). '
        'Selection used the original training and validation groups; fresh confirmation was generated only after model and scale freeze.',
        '',
        '## Decision and measured results',
        '',
        f'The continuous median selected Gaussian sigma **{sigma:g} px = {sigma*.05:g} mm**, '
        f'variance **{(sigma*.05)**2:g} mm²**, from six prespecified choices. '
        'The consensus comparator and the retrained CNN receive the identical smoothed input at this median-selected scale.',
        '',
        '| Method | Pore-task F1 | Pore TP / FP / FN | Pore precision | Pore recall | All-condition F1 | Clean rings flagged / 24 |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for name in LABELS:
        r=result['results'][name];p=r['primary_pores'];a=r['all_conditions']
        text.append(f'| {LABELS[name]} | {p["f1"]:.4f} | {p["tp"]} / {p["fp"]} / {p["fn"]} | '
                    f'{p["precision"]:.4f} | {p["recall"]:.4f} | {a["f1"]:.4f} | {p["clean_rings_flagged"]}/24 |')
    text+=['','Pore-task results include 24 clean and 48 pore scenes. All-condition results additionally include 24 bulges. '
           'A true positive requires one-to-one circular box IoU >= 0.5. F1 = 2TP/(2TP+FP+FN).',
           '','| Paired fresh pore-task difference | F1 difference | 95% background bootstrap interval |',
           '|---|---:|---|']
    for name in ('median','consensus'):
        d=result['cnn_minus_classical'][name]['primary_pores']['f1']
        text.append(f'| CNN minus {name} | {d["difference"]:+.4f} | {d["interval_95"]} |')
    text+=['','Intervals use 2,000 paired background resamples stratified by dwell bin. '
           'They quantify background sampling uncertainty, not uncertainty across CNN training seeds.',
           '',
           '**Interpretation:** the selected smoothed continuous median detects 6/48 pores with 28 false regions, '
           'versus 2/48 with 35 false regions for the retained original median. This is a useful but still weak candidate baseline. '
           'The simple constant-intensity consensus variant does not improve it: 1/48 pores and 60 false regions. '
           'This result applies to this model and fixed tolerance rule, not to every possible RANSAC design.',
           '',
           'The same-scale CNN detects 35/48 pores, but its 117 false regions limit pore F1 to 0.350. '
           'Its paired pore F1 advantage over the median is +0.204, with a wide 95% interval approximately [0.00009, 0.607]. '
           'The lower bound is essentially zero. For all conditions, its F1 advantage is +0.073 with interval [-0.103, 0.442], '
           'so superiority across all defect types is not established by this sample.',
           '',
           'The retained unfiltered CNN performs much better here: 39/48 pores, 11 false regions, pore F1 0.796 and all-condition F1 0.822. '
           'Forced smoothing with retraining at the retained probability threshold did not improve that pipeline. '
           'These are same-confirmation-set comparisons; historical F1 values are not substituted.',
           '',
           '**Recommendation:** retain the original released models and results. Keep the sigma=0.1 mm continuous median '
           'as a separately documented classical baseline candidate, stop this finite search, and do not adopt the tested consensus replacement. '
           'Use the original CNN for the main system comparison and label the shared-blur CNN as a controlled scale ablation. '
           'A smaller CNN remains a future architecture ablation, not a demonstrated improvement.',
           '', '![Validation scale curve](figures/gaussian_consensus_20261020/validation_scale.png)',
           '', '![Fresh confirmation](figures/gaussian_consensus_20261020/confirmation.png)',
           '', '![Matched example](figures/gaussian_consensus_20261020/matched_example.png)',
           '', '## What the method actually does',
           '',
           '1. Unwrap the full circular TOPO band to 160 radial rows by 5,024 angular columns. '
           'Apply the Gaussian to raw TOPO, with angular wrap and reflected radial boundaries, then apply the existing training-only square-root normalization. '
           'The physical blur is isotropic at the nominal 40 mm centreline; angular physical width varies slightly across the radial band. '
           'This is not exactly a native Cartesian Gaussian.',
           '2. At every target pixel, collect eight values at the same radial position, angular offsets '
           '[-256, -192, -128, -64, +64, +128, +192, +256] pixels. The target does not enter the reference. '
           'Nearest references are approximately 3.2 mm away. No profile correction or extra residual blur is used in these new methods.',
           '3. The median reference is the median of those eight values. For the consensus reference, each value proposes a constant intensity. '
           'Count neighbours within the fixed tolerance rule, select the largest agreeing group, break ties by its refitted squared error, '
           'and use the group mean. All eight hypotheses are checked, so this is deterministic exhaustive sample consensus, not random RANSAC.',
           '4. Subtract the reference from the target. Estimate a separate per-row residual centre and noise scale for each method and Gaussian scale '
           'using only clean training scenes: scale = max(0.003, 1.4826 MAD). '
           'Score = abs(residual - centre)/scale. Flag scores >= 12, close with a 3x3 kernel, retain components of at least 15 pixels. '
           'These numeric detection/postprocessing settings are fixed; calibration estimates adapt by a fixed rule.',
           '5. Consensus tolerance is three times the clean-training median residual scale for that radial row. '
           'RANSAC does not remove the need for an agreement tolerance. No additional tolerance or detection threshold search was performed.',
           '6. Choose the median scale by validation pore-task F1, then pixel IoU, then fewer false positives, then smaller sigma. '
           'Freeze before generating confirmation. The retained original patch-based median is included as a separate unchanged control.',
           '',
           '## Finite selection and stopping rule',
           '',
           '| Sigma (pixels) | Sigma (mm) | Variance (mm²) | Median validation pore F1 | Consensus validation pore F1 |',
           '|---:|---:|---:|---:|---:|']
    for s in e.SIGMAS:
        r=selection['trials'][str(s)]
        text.append(f'| {s:g} | {s*.05:g} | {(s*.05)**2:g} | {r["median"]["primary_pores"]["f1"]:.4f} | {r["consensus"]["primary_pores"]["f1"]:.4f} |')
    text += [
        '',
        f'Consensus independently prefers sigma {selection["consensus_secondary_sigma"]:g} px on validation. '
        'That is descriptive only; the main comparison uses the median-selected sigma. '
        'The finite grid is completed once. No enlargement or retuning follows confirmation. '
        'A boundary winner is best among tested scales, not proof of a global optimum.',
        '',
        '## CNN comparison and the supervisor’s smaller network',
        '',
        'The existing architecture is a 262,777-parameter 2D encoder-decoder with skip connections, '
        '64x64 single-TOPO patches and a pixel mask output. Its learned feature widths are 24, 48 and 96. '
        'It is substantially larger than a one- or two-layer 1D CNN.',
        '',
        'This experiment trains that existing architecture from scratch at the chosen scale, '
        'with the original hard-negative recipe: 20 epochs, 1,200 sampled patches per epoch, balanced positive/negative draws, '
        'clean hard-negative mining after epochs 5, 10 and 15, AdamW, cosine learning rate, weighted BCE plus Dice. '
        'Seed 100606002 is fixed in advance. CNN probability threshold 0.35, minimum area 15 and closing 3 are retained. '
        'There is no CNN seed, epoch or threshold selection in this experiment. '
        'The original frozen unfiltered CNN is a separate contextual control, not the same-scale comparator.',
        '',
        'This is a controlled comparison at matched input smoothing and retained operating thresholds. '
        'It is not a comparison of the best achievable settings for every method. '
        'A main system comparison can fairly give both pipelines the same original images while letting Gaussian smoothing '
        'remain part of the classical pipeline and letting the CNN learn its own processing. '
        'The shared-blur experiment answers the narrower question of performance at a common imposed input scale. '
        'The fixed CNN probability threshold may be suboptimal after retraining; no post-result recalibration was performed.',
        '',
        'One concrete interpretation of the supervisor’s proposal is: five adjacent angular columns become five input channels, '
        'each a radial sequence of length 160. Input tensor shape is batch x 5 x 160. '
        'Two Conv1d layers (5 -> 8 and 8 -> 8, kernel length 5, ReLU), followed by a 1x1 output layer, '
        'produce the central column’s defect probabilities. Slide the five-column window around the ring with angular wrap.',
        '',
        'That illustrative architecture has 545 parameters and output shape batch x 1 x 160, verified by an untrained tensor-shape check. '
        'It has a nine-row radial receptive field and five-column angular context before accounting for the Gaussian. '
        'Eight learned feature maps are not the same as eight input measurement channels. '
        'If eight measured profiles are intended, input_channels would instead be eight. '
        'Five columns are context within one example; the training batch dimension is a different axis.',
        '',
        'The small model is easier to interpret and much cheaper, but has far less spatial context and no demonstrated accuracy here. '
        'Treat it as a future capacity ablation using the same groups, preprocessing, masks and metrics. '
        'It was assessed only, not trained or added to this experiment.',
        '',
        '## Acquisition-noise and missed-pore diagnostics',
        '',
        'Dwell time is an acquisition-noise proxy in this simulator, not a measured physical noise-power calibration. '
        'Each bin contains only six independent backgrounds, so subgroup differences are descriptive.',
        '',
        '| Dwell interval (microseconds) | Median pore F1 | Consensus pore F1 | Same-scale CNN pore F1 | Same-scale CNN false regions |',
        '|---|---:|---:|---:|---:|',
    ]
    for b,(lo,hi) in enumerate(zip(e.original.EDGES[:-1],e.original.EDGES[1:])):
        values=[]
        for name in ('median','consensus','cnn_same_scale'):
            rows=[r for r in result['results'][name]['rows'] if r['dwell_bin']==b and r['source_class']!='bulge']
            values.append(e.metrics.summary(rows)['f1'])
        text.append(f'| {lo:g} to {hi:g} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {sum(r["fp"] for r in rows)} |')
    text += [
        '',
        'The same-scale CNN makes 114 of its 117 pore-task false-region detections in the shortest dwell bin (0.1 to 0.5 microseconds). '
        'Its false-alarm problem is concentrated in that acquisition regime at the retained operating point. '
        'This locates the observed failure; it does not establish a calibrated physical noise-power cause.',
        '',
        '| Method | Matched pores | Unmatched pores with some predicted-pixel overlap | Pores with no predicted-pixel overlap |',
        '|---|---:|---:|---:|',
    ]
    for name in ('median','consensus','cnn_same_scale'):
        rows=[r for r in result['results'][name]['rows'] if r['source_class'] in ('open_keyhole','open_intrinsic')]
        matched=sum(r['tp'] for r in rows)
        partial=sum(r['tp']==0 and r['intersection']>0 for r in rows)
        absent=sum(r['tp']==0 and r['intersection']==0 for r in rows)
        assert matched+partial+absent==48
        text.append(f'| {LABELS[name]} | {matched} | {partial} | {absent} |')
    text += [
        '',
        'A partial overlap can represent a response to the pore that still fails the required box localisation. '
        'No pixel overlap means the final thresholded, area-filtered prediction misses the pore; it does not prove the raw image contains no signal.',
        '',
        '## Interpretation limits',
        '',
        '- Gaussian smoothing suppresses high-frequency noise but can also suppress small pores and spread boundaries.',
        '- The median already resists a minority of extreme neighbouring samples. RANSAC targets outliers inconsistent with a model; '
        'neither method guarantees robustness to arbitrary noise power or correlated weld variation.',
        '- Smooth observations still support pointwise output. The issue is scale, model adequacy and evidence, not that all pointwise statistics are inherently non-robust.',
        '- Only Gaussian scale is searched; training-derived calibrations and CNN weights change according to fixed rules. '
        'This is not a test with every numerical value literally unchanged.',
        '- This comparison concerns synthetic rendered images with declared ring coordinates, not physical manufacturing inspection or unknown-image localisation.',
        '- The illustration uses the prespecified first fresh background, not a visually selected success case.',
        '',
        '## Reproduction and evidence',
        '',
        '`WeldDefectDetection/run_gaussian_consensus_20261020.py all` runs the staged experiment with the repository Python environment. '
        '`WeldDefectDetection/report_gaussian_consensus_20261020.py` rebuilds this report and figures. '
        'Protocol, numerical checks, per-scale validation rows, training checkpoint, frozen hashes, fresh scene records, per-scene predictions, '
        'paired intervals and audit are in `DataSimulation/data/output/gaussian_consensus_20261020_v1`.',
        '',
        'Sources: [scikit-image RANSAC definition](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac), '
        '[SciPy Gaussian sigma definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html), '
        '[PyTorch Conv1d tensor convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).'
    ]
    DOC.write_text('\n'.join(text)+'\n',encoding='utf-8')
    e.write(FIG/'provenance.json',dict(campaign=str(e.OUT),freeze_sha256=e.sha(e.OUT/'frozen.json'),
        results_sha256=e.sha(e.OUT/'results.json'),figure_script_sha256=e.sha(Path(__file__)),
        example_background=0,case_selection='Prespecified first fresh background, clean and open_intrinsic',
        verification=verification))
    print(DOC)


if __name__=='__main__':main()
